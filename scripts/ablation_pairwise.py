#!/usr/bin/env python3
"""
Ablation Study Nhóm 3 (Run 9): XGBoost Pairwise
So sánh thuật toán học theo cặp (Pairwise) với Baseline học theo danh sách (NDCG).
"""

import os, sys, pickle
import numpy as np, xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

def calculate_f2(gt_ids, pred_ids):
    gts, preds = set(gt_ids), set(pred_ids)
    if not gts: return 0.0
    tp = len(preds & gts)
    fp = len(preds - gts)
    fn = len(gts - preds)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    return (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0

def run_pairwise_ablation():
    cache_path = DATA_DIR / "features_cache_5f_with_text.pkl"
    with open(cache_path, 'rb') as f:
        X_train, y_train, qids, test_processed = pickle.load(f)

    print("🚀 Đang huấn luyện XGBoost với thuật toán Pairwise...")
    # ĐIỂM KHÁC BIỆT NẰM Ở ĐÂY: objective="rank:pairwise"
    ranker = xgb.XGBRanker(
        n_jobs=-1, tree_method="hist", objective="rank:pairwise",
        n_estimators=150, learning_rate=0.05, max_depth=6
    )
    ranker.fit(X_train, y_train, qid=qids)

    MARGIN = 0.7
    f2_list = []
    
    for item in test_processed:
        f_scores = ranker.predict(item["X"].values) # Chuyển DF thành numpy array
        
        d = {}
        for art, s in zip(item["map"], f_scores):
            if art not in d or s > d[art]: d[art] = s
            
        art_list = sorted(d.items(), key=lambda x: x[1], reverse=True)
        top_score = art_list[0][1]
        preds = [a for a, s in art_list if top_score - s <= MARGIN]
        f2_list.append(calculate_f2(item["gt"], preds))

    final_f2 = np.mean(f2_list)
    
    print("\n" + "="*50)
    print("🏆 KẾT QUẢ ABLATION STUDY: XGBOOST PAIRWISE 🏆")
    print("="*50)
    print(f"Thuật toán : rank:pairwise")
    print(f"F2-Macro   : {final_f2:.6f}")
    print(f"Độ sụt giảm: {0.8772 - final_f2:+.4f} (so với NDCG)")
    print("="*50)

if __name__ == "__main__":
    run_pairwise_ablation()