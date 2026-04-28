import json
import os
import sys
import string
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
from tqdm import tqdm
from sentence_transformers import CrossEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval.hybrid import MultiViewHybridRetriever
from config import TRAIN_FILE, MODELS_DIR

def calculate_lexical_features(query, chunk_text):
    """Tính toán đặc trưng từ vựng (Lexical Overlap)"""
    q_clean = query.lower().translate(str.maketrans('', '', string.punctuation))
    c_clean = chunk_text.lower().translate(str.maketrans('', '', string.punctuation))
    
    q_words = set(q_clean.split())
    c_words = set(c_clean.split())
    
    if not q_words:
        return 0.0, 0.0
        
    overlap_count = len(q_words.intersection(c_words))
    overlap_ratio = overlap_count / len(q_words)
    
    legal_keywords = {"điều", "khoản", "phạt", "tù", "bồi thường", "cấm", "nghị định", "luật", "quy định", "trách nhiệm"}
    keyword_overlap = len(q_words.intersection(c_words).intersection(legal_keywords))
    
    return float(overlap_ratio), float(keyword_overlap)

def train_xgboost_with_lexical():
    print("🚀 BƯỚC 1: Khởi động hệ thống trích xuất đặc trưng...")
    retriever = MultiViewHybridRetriever(load_models=True)
    ce_model = CrossEncoder(str(MODELS_DIR / "reranker_model"), device="cuda:0")

    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    X_train_list = []
    y_train_list = []
    qids = []

    TOP_K_RETRIEVAL = 30
    features_order = ["s_hybrid", "s_ce", "p_ce", "u_uncertainty", "s_gat", "f_overlap", "f_key_match"]

    print(f"\n🧠 BƯỚC 2: Trích xuất 7 Features trên {len(train_data)} câu hỏi Train...")
    for qid, item in enumerate(tqdm(train_data)):
        query = item.get("text", "")
        gt_ids = [f"{rel['law_id']}_{rel['article_id']}" for rel in item.get("relevant_articles", [])]
        if not gt_ids: continue

        scores = retriever._get_combined_scores(query)
        top_indices = np.argsort(scores)[::-1][:TOP_K_RETRIEVAL]
        
        pairs = [[query, retriever.chunks[idx]['text']] for idx in top_indices]
        ce_raw_scores = ce_model.predict(pairs, batch_size=32, convert_to_numpy=True)
        ce_probs = torch.nn.functional.softmax(torch.tensor(ce_raw_scores), dim=0).numpy()
        
        sorted_probs = np.sort(ce_probs)[::-1]
        u_val = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0

        gat_norm = retriever._get_graph_scores(query)
        
        for i, idx in enumerate(top_indices):
            chunk = retriever.chunks[idx]
            art_key = f"{chunk['law_id']}_{chunk['article_id']}"
            
            # Tính điểm Lexical
            f_overlap, f_key = calculate_lexical_features(query, chunk['text'])
            gat_score = gat_norm[retriever.gat_mapping[art_key]] if art_key in retriever.gat_mapping else 0.0
            
            X_train_list.append([
                float(scores[idx]),          # s_hybrid
                float(ce_raw_scores[i]),     # s_ce
                float(ce_probs[i]),          # p_ce
                float(u_val),                # u_uncertainty
                float(gat_score),            # s_gat
                f_overlap,                   # f_overlap (MỚI)
                f_key                        # f_key_match (MỚI)
            ])
            y_train_list.append(1 if art_key in gt_ids else 0)
            qids.append(qid)

    print("\n🚀 BƯỚC 3: Huấn luyện XGBRanker với Objective rank:ndcg...")
    X_train = pd.DataFrame(X_train_list, columns=features_order)
    y_train = np.array(y_train_list)
    
    ranker = xgb.XGBRanker(
        tree_method="hist",
        objective="rank:ndcg",
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8
    )
    ranker.fit(X_train, y_train, qid=qids)

    # Lưu Model
    out_model = MODELS_DIR / "xgboost_lexical_ranker.json"
    ranker.save_model(str(out_model))
    print(f"✅ Đã lưu mô hình XGBoost mới tại: {out_model}")

if __name__ == "__main__":
    train_xgboost_with_lexical()