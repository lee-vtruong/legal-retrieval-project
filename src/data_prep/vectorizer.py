import json
import pickle
import numpy as np
import os
import sys
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

# Import từ config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import CHUNK_FILE, EMBEDDINGS_DIR, DEVICE

class VectorBuilder:
    def __init__(self):
        print(f"Reading chunks from {CHUNK_FILE}...")
        with open(CHUNK_FILE, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        self.texts = [chunk["text"] for chunk in self.chunks]
        print(f"Loaded {len(self.texts)} chunks for vectorization.")

    def build_sparse_indices(self):
        """Xây dựng bộ chỉ mục từ vựng (Lexical)"""
        print("\n--- Building Sparse Indices ---")
        
        # 1. BM25
        print("Building BM25 Index...")
        tokenized_corpus = [doc.lower().split() for doc in self.texts]
        bm25 = BM25Okapi(tokenized_corpus)
        with open(EMBEDDINGS_DIR / "bm25_index.pkl", "wb") as f:
            pickle.dump(bm25, f)
            
        # 2. TF-IDF
        print("Building TF-IDF Index...")
        tfidf = TfidfVectorizer(lowercase=True)
        tfidf_matrix = tfidf.fit_transform(self.texts)
        with open(EMBEDDINGS_DIR / "tfidf_index.pkl", "wb") as f:
            pickle.dump({"vectorizer": tfidf, "matrix": tfidf_matrix}, f)
            
        print("Sparse indices saved successfully!")

    def build_dense_embeddings(self):
        """Xây dựng vector ngữ nghĩa (Semantic)"""
        print("\n--- Building Dense Embeddings ---")
        models_to_run = {
            "bge_m3": "BAAI/bge-m3",
            "vietnamese": "AITeamVN/Vietnamese_Embedding_v2",
            "e5_large": "intfloat/multilingual-e5-large"
        }

        for name, model_path in models_to_run.items():
            print(f"\nLoading Dense Model: {model_path} on {DEVICE}...")
            model = SentenceTransformer(model_path, device=DEVICE)
            
            # E5 yêu cầu tiền tố "passage: " cho văn bản đầu vào để tối ưu hóa
            texts_to_encode = self.texts
            if "e5" in name.lower():
                texts_to_encode = [f"passage: {t}" for t in self.texts]
            
            print(f"Encoding {len(texts_to_encode)} chunks with {name}...")
            embeddings = model.encode(texts_to_encode, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
            
            save_path = EMBEDDINGS_DIR / f"{name}_embeddings.npy"
            np.save(save_path, embeddings)
            print(f"Saved {name} embeddings to {save_path}")

    def run(self):
        self.build_sparse_indices()
        self.build_dense_embeddings()
        print("\nVectorization Completed Successfully! 🚀")