#!/usr/bin/env python3
"""
Ablation study λ (Sparse / Dense / GAT) trên 82 câu test (Recall@30).
"""

import numpy as np
import pickle
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from FlagEmbedding import BGEM3FlagModel
from config import TEST_FILE, DATA_DIR
from src.retrieval.hybrid import MultiViewHybridRetriever

def min_max_normalize(scores):
    min_v, max_v = np.min(scores), np.max(scores)
    if max_v == min_v:
        return np.full_like(scores, 0.5)
    return (scores - min_v) / (max_v - min_v)

def calculate_recall_at_k(gt_ids, ranked_candidates, k=30):
    preds_k = set(ranked_candidates[:k])
    gts = set(gt_ids)
    if not gts:
        return 0.0
    tp = len(preds_k & gts)
    return tp / len(gts)

def main():
    print("🚀 BƯỚC 1: LOAD TÀI NGUYÊN (INDEX, EMBEDDINGS, CHUNKS)...")

    # 1. BM25 (chunk level)
    with open(f"{DATA_DIR}/embeddings/bm25_index.pkl", "rb") as f:
        bm25_model = pickle.load(f)

    # 2. Dense & GAT Embeddings
    dense_embs = np.load(f"{DATA_DIR}/embeddings/bge_m3_embeddings.npy")
    gat_embs = np.load(f"{DATA_DIR}/embeddings/gat_embeddings.npy")

    # 3. GAT node mapping → list of article_id strings
    with open(f"{DATA_DIR}/embeddings/gat_node_mapping.json", "r") as f:
        gat_mapping_raw = json.load(f)

    article_ids_gat = []
    if isinstance(gat_mapping_raw, list):
        for item in gat_mapping_raw:
            if isinstance(item, dict):
                article_ids_gat.append(f"{item['law_id']}_{item['article_id']}")
            else:
                article_ids_gat.append(str(item))
    elif isinstance(gat_mapping_raw, dict):
        article_ids_gat = [gat_mapping_raw[str(i)] for i in range(len(gat_mapping_raw))]
    else:
        raise TypeError("gat_node_mapping.json phải là list hoặc dict")

    n_articles = len(article_ids_gat)
    print(f"✅ Số article trong đồ thị GAT: {n_articles}")

    # 4. Lấy thông tin chunks (chỉ metadata, không load model nặng)
    print("📚 Đọc chunks từ Hybrid Retriever...")
    retriever = MultiViewHybridRetriever(load_models=False)
    chunks = retriever.chunks
    chunk_article_ids = [f"{c['law_id']}_{c['article_id']}" for c in chunks]
    n_chunks = len(chunks)
    print(f"✅ Số chunk: {n_chunks}")

    # Kiểm tra khớp BM25 & Dense
    test_bm25_scores = bm25_model.get_scores("test".split())
    assert len(test_bm25_scores) == n_chunks, \
        f"BM25 index size {len(test_bm25_scores)} không khớp số chunk {n_chunks}"
    assert dense_embs.shape[0] == n_chunks, \
        f"Dense embeddings shape {dense_embs.shape[0]} không khớp số chunk {n_chunks}"

    # 5. Load Query Encoder
    print("🧠 Đang nạp mô hình BGE-M3...")
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    # 6. Load tập test
    print(f"📂 Đang nạp tập test từ: {TEST_FILE}")
    test_queries = []
    if str(TEST_FILE).endswith('.jsonl'):
        with open(TEST_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                test_queries.append(json.loads(line))
    else:
        with open(TEST_FILE, 'r', encoding='utf-8') as f:
            test_queries = json.load(f)

    for q in test_queries:
        if 'gt_ids' not in q and 'relevant_articles' in q:
            q['gt_ids'] = [f"{r['law_id']}_{r['article_id']}" for r in q['relevant_articles']]
        if 'query_text' not in q and 'text' in q:
            q['query_text'] = q['text']

    print(f"✅ Đã nạp {len(test_queries)} câu hỏi test.")

    # 7. Tính điểm thô và aggregate lên article
    print("\n⚙️ BƯỚC 2: TÍNH ĐIỂM THÔ VÀ GOM VỀ ARTICLE...")
    article_scores_list = []

    for q in tqdm(test_queries, desc="Tính điểm"):
        text = q.get('query_text', q.get('question', ''))
        gt_ids = q.get('gt_ids', [])

        tokenized = text.lower().split()
        s_sparse_chunk = bm25_model.get_scores(tokenized)          # (n_chunks,)
        q_dense_emb = model.encode([text])['dense_vecs']
        s_dense_chunk = cosine_similarity(q_dense_emb, dense_embs)[0]  # (n_chunks,)

        # Max pooling lên article
        max_sparse = {}
        max_dense = {}
        for idx, art_id in enumerate(chunk_article_ids):
            ss = s_sparse_chunk[idx]
            sd = s_dense_chunk[idx]
            if art_id not in max_sparse or ss > max_sparse[art_id]:
                max_sparse[art_id] = ss
            if art_id not in max_dense or sd > max_dense[art_id]:
                max_dense[art_id] = sd

        # GAT ở cấp article
        s_gat_article = cosine_similarity(q_dense_emb, gat_embs)[0]  # (n_articles,)

        # Sắp theo thứ tự article_ids_gat
        sparse_vec = np.zeros(n_articles)
        dense_vec = np.zeros(n_articles)
        for i, art_id in enumerate(article_ids_gat):
            sparse_vec[i] = max_sparse.get(art_id, 0.0)
            dense_vec[i] = max_dense.get(art_id, 0.0)

        article_scores_list.append({
            'gt': gt_ids,
            'sparse': sparse_vec,
            'dense': dense_vec,
            'gat': s_gat_article
        })

    # 8. Grid search λ
    print("\n🔬 BƯỚC 3: CHẠY GRID SEARCH CHO LAMBDA...")
    steps = [round(x * 0.1, 1) for x in range(11)]
    valid_lambdas = [(l1, l2, round(1.0 - l1 - l2, 1))
                     for l1 in steps for l2 in steps
                     if 0.0 <= round(1.0 - l1 - l2, 1) <= 1.0]

    results = []
    for l1, l2, l3 in tqdm(valid_lambdas, desc="Quét λ"):
        recalls = []
        for item in article_scores_list:
            norm_sparse = min_max_normalize(item['sparse'])
            norm_dense = min_max_normalize(item['dense'])
            norm_gat = min_max_normalize(item['gat'])

            s_mv = l1 * norm_sparse + l2 * norm_dense + l3 * norm_gat

            scored = list(zip(article_ids_gat, s_mv))
            scored.sort(key=lambda x: x[1], reverse=True)
            top30 = [c[0] for c in scored[:30]]
            recalls.append(calculate_recall_at_k(item['gt'], top30, k=30))
        results.append((l1, l2, l3, np.mean(recalls)))

    results.sort(key=lambda x: x[3], reverse=True)

    print("\n🏆 KẾT QUẢ TỐI ƯU NHẤT (TOP 5):")
    print(f"{'λ1 (Sparse)':<15} | {'λ2 (Dense)':<15} | {'λ3 (GAT)':<15} | {'Recall@30':<10}")
    print("-" * 65)
    for l1, l2, l3, rec in results[:5]:
        print(f"{l1:<15.1f} | {l2:<15.1f} | {l3:<15.1f} | {rec:.4f}")

if __name__ == "__main__":
    main()