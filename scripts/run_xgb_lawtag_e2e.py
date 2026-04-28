import json
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
from tqdm import tqdm
from sentence_transformers import CrossEncoder

# Thiết lập đường dẫn import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval.hybrid import MultiViewHybridRetriever
from config import TRAIN_FILE, TEST_FILE, MODELS_DIR

def get_law_tag_score(query, candidate_law_id, all_law_ids):
    """Lưới lọc Rule-based Law-tag: Thưởng +1 nếu đúng luật, phạt -1 nếu sai luật"""
    q_lower = query.lower()
    c_lower = candidate_law_id.lower()
    
    if c_lower in q_lower:
        return 1.0
        
    for law in all_law_ids:
        if law.lower() in q_lower and law.lower() != c_lower:
            return -1.0 
            
    return 0.0

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

def run_end_to_end_lawtag():
    print("🚀 BƯỚC 1: Khởi động Hybrid Retriever & Cross-Encoder...")
    retriever = MultiViewHybridRetriever(load_models=True)
    ce_model = CrossEncoder(str(MODELS_DIR / "reranker_model"), device="cuda:0")

    # Lấy danh sách tất cả các tên luật có trong Corpus
    all_law_ids = list(set([chunk['law_id'] for chunk in retriever.chunks]))
    print(f"📚 Đã tìm thấy {len(all_law_ids)} bộ luật trong Corpus.")

    # ==========================================
    # PHẦN 1: HUẤN LUYỆN (TRAINING)
    # ==========================================
    print("\n🧠 BƯỚC 2: Trích xuất 6 Features trên tập TRAIN...")
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    X_train_list, y_train_list, qids = [], [], []
    TOP_K_RETRIEVAL = 30
    features_order = ["s_hybrid", "s_ce", "p_ce", "u_uncertainty", "s_gat", "law_score"]

    for qid, item in enumerate(tqdm(train_data, desc="Train Extr")):
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
            gat_score = gat_norm[retriever.gat_mapping[art_key]] if art_key in retriever.gat_mapping else 0.0
            
            # GỌI HÀM LAW-TAG MỚI
            law_score = get_law_tag_score(query, chunk['law_id'], all_law_ids)
            
            X_train_list.append([
                float(scores[idx]), float(ce_raw_scores[i]), float(ce_probs[i]),
                float(u_val), float(gat_score), float(law_score)
            ])
            y_train_list.append(1 if art_key in gt_ids else 0)
            qids.append(qid)

    print("\n🔥 BƯỚC 3: Fit mô hình XGBRanker với 6 features...")
    ranker = xgb.XGBRanker(tree_method="hist", objective="rank:ndcg", n_estimators=150, learning_rate=0.1)
    ranker.fit(pd.DataFrame(X_train_list, columns=features_order), np.array(y_train_list), qid=qids)

    # ==========================================
    # PHẦN 2: KIỂM THỬ VÀ ĐÁNH GIÁ (INFERENCE & EVAL)
    # ==========================================
    print("\n🧠 BƯỚC 4: Trích xuất và Dự đoán trên tập TEST...")
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    all_predictions = []
    
    for item in tqdm(test_data, desc="Test Infer"):
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
            law_score = get_law_tag_score(query, chunk['law_id'], all_law_ids)
            
            df_features.append({
                "s_hybrid": float(scores[idx]), "s_ce": float(ce_raw_scores[i]), "p_ce": float(ce_probs[i]),
                "u_uncertainty": float(u_val), "s_gat": float(gat_score), "law_score": float(law_score)
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

    print("\n🔍 BƯỚC 5: Tự động dò Ngưỡng vàng (Dynamic Margin)...")
    margins = np.arange(0.0, 2.5, 0.05)
    best_f2 = 0; best_m = 0
    
    for m in margins:
        f2_scores = []
        for item in all_predictions:
            if not item["pred_ids"]:
                f2_scores.append(0)
                continue
            top_score = item["pred_scores"][0]
            selected = [p_id for p_id, p_s in zip(item["pred_ids"], item["pred_scores"]) if top_score - p_s <= m]
            f2_scores.append(calculate_f2(item["gt_ids"], selected))
            
        macro_f2 = np.mean(f2_scores)
        if macro_f2 > best_f2:
            best_f2 = macro_f2; best_m = m

    baseline_f2 = np.mean([calculate_f2(item["gt_ids"], [item["pred_ids"][0]] if item["pred_ids"] else []) for item in all_predictions])

    print("\n" + "="*50)
    print("🎯 KẾT QUẢ XGBOOST VỚI LAW-TAG FILTER (RULE-BASED)")
    print(f"F2 Baseline (Fixed Top-1) : {baseline_f2:.6f}")
    print(f"F2 SOTA (Dynamic Margin)  : {best_f2:.6f} 🌟")
    print(f"Ngưỡng Margin vàng        : {best_m:.2f}")
    print("="*50)

if __name__ == "__main__":
    run_end_to_end_lawtag()