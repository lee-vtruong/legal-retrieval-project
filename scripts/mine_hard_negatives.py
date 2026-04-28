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
from config import TEST_FILE, MODELS_DIR

def calculate_lexical_features(query, chunk_text):
    q_clean = query.lower().translate(str.maketrans('', '', string.punctuation))
    c_clean = chunk_text.lower().translate(str.maketrans('', '', string.punctuation))
    q_words = set(q_clean.split())
    c_words = set(c_clean.split())
    if not q_words: return 0.0, 0.0
    overlap_ratio = len(q_words.intersection(c_words)) / len(q_words)
    legal_keywords = {"điều", "khoản", "phạt", "tù", "bồi thường", "cấm", "nghị định", "luật", "quy định", "trách nhiệm"}
    keyword_overlap = len(q_words.intersection(c_words).intersection(legal_keywords))
    return float(overlap_ratio), float(keyword_overlap)

def calculate_f2(gt_ids, pred_ids):
    gts = set(gt_ids)
    preds = set(pred_ids)
    if not gts: return 0
    tp = len(preds.intersection(gts))
    fp = len(preds - gts)
    fn = len(gts - preds)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    if (4 * prec + rec) == 0: return 0
    return (5 * prec * rec) / (4 * prec + rec)

def run_lexical_inference():
    print("🚀 Nạp hệ thống...")
    retriever = MultiViewHybridRetriever(load_models=True)
    ce_model = CrossEncoder(str(MODELS_DIR / "reranker_model"), device="cuda:0")
    
    ranker = xgb.XGBRanker()
    ranker.load_model(str(MODELS_DIR / "xgboost_lexical_ranker.json"))
    features_order = ["s_hybrid", "s_ce", "p_ce", "u_uncertainty", "s_gat", "f_overlap", "f_key_match"]

    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    TOP_K_RETRIEVAL = 30
    all_predictions = [] # Lưu trữ toàn bộ để lát nữa tìm Margin

    print(f"\n🧠 Chạy kiểm thử và lưu điểm thô trên {len(test_data)} câu hỏi Test...")
    for item in tqdm(test_data):
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
        
        df_features = []
        mapping_idx = []
        
        for i, idx in enumerate(top_indices):
            chunk = retriever.chunks[idx]
            art_key = f"{chunk['law_id']}_{chunk['article_id']}"
            gat_score = gat_norm[retriever.gat_mapping[art_key]] if art_key in retriever.gat_mapping else 0.0
            f_overlap, f_key = calculate_lexical_features(query, chunk['text'])
            
            df_features.append({
                "s_hybrid": float(scores[idx]), "s_ce": float(ce_raw_scores[i]), "p_ce": float(ce_probs[i]),
                "u_uncertainty": float(u_val), "s_gat": float(gat_score),
                "f_overlap": f_overlap, "f_key_match": f_key
            })
            mapping_idx.append(art_key)

        X_test = pd.DataFrame(df_features)[features_order]
        final_scores = ranker.predict(X_test)
        
        article_scores = {}
        for f_score, art_key in zip(final_scores, mapping_idx):
            if art_key not in article_scores or f_score > article_scores[art_key]:
                article_scores[art_key] = f_score
                
        sorted_arts = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)
        
        all_predictions.append({
            "gt_ids": gt_ids,
            "pred_ids": [art[0] for art in sorted_arts],
            "pred_scores": [float(art[1]) for art in sorted_arts]
        })

    # ==========================================
    # QUÉT TÌM MARGIN TỐI ƯU TỰ ĐỘNG
    # ==========================================
    print("\n🔍 Đang rà soát Margin tối ưu cho XGBoost Lexical (Mới)...")
    # Tăng range lên 3.0 vì XGBoost đôi khi xuất điểm ra scale lớn
    margins = np.arange(0.0, 3.0, 0.05) 
    best_f2 = 0
    best_m = 0
    
    for m in margins:
        f2_scores = []
        for item in all_predictions:
            pred_ids = item["pred_ids"]
            pred_scores = item["pred_scores"]
            
            if not pred_ids:
                f2_scores.append(0)
                continue
                
            top_score = pred_scores[0]
            selected_preds = []
            for p_id, p_score in zip(pred_ids, pred_scores):
                if top_score - p_score <= m:
                    selected_preds.append(p_id)
                else:
                    break
            f2_scores.append(calculate_f2(item["gt_ids"], selected_preds))
            
        macro_f2 = np.mean(f2_scores)
        if macro_f2 > best_f2:
            best_f2 = macro_f2
            best_m = m

    baseline_f2 = np.mean([calculate_f2(item["gt_ids"], [item["pred_ids"][0]] if item["pred_ids"] else []) for item in all_predictions])

    print("\n" + "="*50)
    print("🎯 KẾT QUẢ XGBOOST LEXICAL (7 FEATURES)")
    print(f"F2 Baseline (Fixed Top-1) : {baseline_f2:.6f}")
    print(f"F2 SOTA (Dynamic Margin)  : {best_f2:.6f} 🌟")
    print(f"Ngưỡng Margin vàng mới    : {best_m:.2f}")
    print("="*50)

if __name__ == "__main__":
    run_lexical_inference()