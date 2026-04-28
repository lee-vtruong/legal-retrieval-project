import json
import pickle
import numpy as np
import torch
import os
import sys
from sentence_transformers import SentenceTransformer

# Import từ config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import CHUNK_FILE, EMBEDDINGS_DIR, DEVICE, DATA_DIR

def min_max_normalize(scores):
    """Chuẩn hóa điểm số về khoảng [0, 1] để fusion"""
    min_val, max_val = np.min(scores), np.max(scores)
    if max_val - min_val == 0:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)

class MultiViewHybridRetriever:
    def __init__(self, load_models=True):
        print("🚀 Khởi tạo Stage 2 & 3: Multi-View Hybrid + GAT Retriever...")
        
        # 1. Nạp danh sách văn bản (Chunks)
        with open(CHUNK_FILE, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
            
        # 2. Nạp Sparse Index (Stage 2 - Lexical)
        print("Nạp BM25 và TF-IDF index...")
        with open(EMBEDDINGS_DIR / "bm25_index.pkl", 'rb') as f:
            self.bm25 = pickle.load(f)
        with open(EMBEDDINGS_DIR / "tfidf_index.pkl", 'rb') as f:
            tfidf_data = pickle.load(f)
            self.tfidf_vectorizer = tfidf_data["vectorizer"]
            self.tfidf_matrix = tfidf_data["matrix"]
            
        # 3. Nạp Dense Vectors (Stage 2 - Semantic)
        print("Nạp Dense Embeddings (BGE, Viet, E5)...")
        self.emb_bge = np.load(EMBEDDINGS_DIR / "bge_m3_embeddings.npy")
        self.emb_vie = np.load(EMBEDDINGS_DIR / "vietnamese_embeddings.npy")
        self.emb_e5 = np.load(EMBEDDINGS_DIR / "e5_large_embeddings.npy")
        
        # 4. Nạp GAT Embeddings (Stage 3 - Topological)
        print("Nạp GAT Embeddings và Node Mapping...")
        self.gat_emb = np.load(EMBEDDINGS_DIR / "gat_embeddings.npy")
        with open(EMBEDDINGS_DIR / "gat_node_mapping.json", "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
            # Tạo dictionary để tra cứu nhanh index của article trong ma trận GAT
            self.gat_mapping = {f"{item['law_id']}_{item['article_id']}": idx for idx, item in enumerate(mapping_data)}

        # 5. Nạp Models (để encode câu query trực tiếp)
        if load_models:
            print(f"Nạp các mô hình Transformer lên {DEVICE}...")
            self.model_bge = SentenceTransformer("BAAI/bge-m3", device=DEVICE)
            self.model_vie = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", device=DEVICE)
            self.model_e5 = SentenceTransformer("intfloat/multilingual-e5-large", device=DEVICE)
            
        print("✅ Hệ thống Hybrid + GAT sẵn sàng!")

    def _get_sparse_scores(self, query):
        """Chấm điểm Lexical (BM25, TF-IDF)"""
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        query_vec = self.tfidf_vectorizer.transform([query])
        tfidf_scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()
        
        return min_max_normalize(bm25_scores), min_max_normalize(tfidf_scores)

    def _get_dense_scores(self, query):
        """Chấm điểm Semantic (BGE, Viet, E5)"""
        q_bge = self.model_bge.encode([query], normalize_embeddings=True)
        q_vie = self.model_vie.encode([query], normalize_embeddings=True)
        q_e5 = self.model_e5.encode([f"query: {query}"], normalize_embeddings=True)
        
        score_bge = np.dot(self.emb_bge, q_bge.T).flatten()
        score_vie = np.dot(self.emb_vie, q_vie.T).flatten()
        score_e5 = np.dot(self.emb_e5, q_e5.T).flatten()
        
        return min_max_normalize(score_bge), min_max_normalize(score_vie), min_max_normalize(score_e5)

    def _get_graph_scores(self, query):
        """Chấm điểm dựa trên tương đồng vector GAT (Topological)"""
        # Sử dụng BGE model để encode vì GAT được train dựa trên features của BGE
        q_emb = self.model_bge.encode([query], normalize_embeddings=True)
        gat_scores_raw = np.dot(self.gat_emb, q_emb.T).flatten()
        return min_max_normalize(gat_scores_raw)

    def _get_combined_scores(self, query):
        """
        Phương thức nội bộ để lấy điểm số tổng hợp từ tất cả các view 
        phục vụ cho việc chạy Ablation Study.
        """
        # 1. Lấy điểm từ các thành phần
        bm25_s, tfidf_s = self._get_sparse_scores(query)
        bge_s, vie_s, e5_s = self._get_dense_scores(query)
        gat_norm = self._get_graph_scores(query)

        # 2. Map điểm GAT từ Article/Law về từng Chunk
        full_gat_scores = np.zeros(len(self.chunks))
        for i, chunk in enumerate(self.chunks):
            key = f"{chunk['law_id']}_{chunk['article_id']}"
            if key in self.gat_mapping:
                full_gat_scores[i] = gat_norm[self.gat_mapping[key]]

        # 3. Hợp nhất điểm theo trọng số Proposed của Stage 2 & 3
        # Trọng số: [bge: 0.3, bm25: 0.2, gat: 0.2, vie: 0.1, e5: 0.1, tfidf: 0.1]
        combined_scores = (0.3 * bge_s) + (0.2 * bm25_s) + (0.2 * full_gat_scores) + \
                          (0.1 * vie_s) + (0.1 * e5_s) + (0.1 * tfidf_s)
        
        return combined_scores

    def retrieve(self, query, top_k=60, return_all_scores=False):
        """Truy xuất tài liệu bằng cách Fusion tín hiệu từ Stage 2 và Stage 3"""
        
        # 1. Lấy điểm Stage 2 (Sparse & Dense)
        bm25_s, tfidf_s = self._get_sparse_scores(query)
        bge_s, vie_s, e5_s = self._get_dense_scores(query)
        
        # 2. Lấy điểm Stage 3 (GAT)
        gat_scores_norm = self._get_graph_scores(query)
        
        # Ánh xạ điểm GAT từ Article xuống các Chunks tương ứng
        full_gat_scores = np.zeros(len(self.chunks))
        for idx, chunk in enumerate(self.chunks):
            key = f"{chunk['law_id']}_{chunk['article_id']}"
            if key in self.gat_mapping:
                full_gat_scores[idx] = gat_scores_norm[self.gat_mapping[key]]

        # 3. Fusion Điểm số (Ensemble Weights) theo paper GAM-Hybrid
        # Kết hợp ưu thế của BGE, BM25 và sức mạnh dẫn chiếu của GAT
        final_scores = (0.30 * bge_s) + \
                       (0.20 * bm25_s) + \
                       (0.20 * full_gat_scores) + \
                       (0.10 * vie_s) + \
                       (0.10 * e5_s) + \
                       (0.10 * tfidf_s)
        
        # Lấy top_k ứng viên
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "law_id": chunk["law_id"],
                "article_id": chunk["article_id"],
                "text": chunk["text"],
                "level": chunk["level"],
                "hybrid_score": float(final_scores[idx]),
                "score_components": {
                    "bm25": float(bm25_s[idx]),
                    "bge": float(bge_s[idx]),
                    "gat": float(full_gat_scores[idx]),
                    "vie": float(vie_s[idx]),
                    "e5": float(e5_s[idx]),
                    "tfidf": float(tfidf_s[idx])
                }
            })
            
        if return_all_scores:
            return results, final_scores
        return results