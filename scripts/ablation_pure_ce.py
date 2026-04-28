#!/usr/bin/env python3
"""
Ablation Study Nhóm 3: Pure Fine-tuned Cross-Encoder
Đánh giá sức mạnh của riêng mô hình CE (không có XGBoost Meta-Learner).
"""

import os, sys, pickle
import numpy as np
from tqdm import tqdm

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

def run_pure_ce_ablation():
    cache_path = DATA_DIR / "features_cache_5f_with_text.pkl"
    with open(cache_path, 'rb') as f:
        _, _, _, test_processed = pickle.load(f)

    # Trong 5 features: ["s_hybrid", "s_ce", "p_ce", "u_uncertainty", "s_gat"]
    # Điểm thô của Cross-Encoder nằm ở cột thứ 2 (index 1)
    # Xác suất (Softmax) của Cross-Encoder nằm ở cột thứ 3 (index 2)
    
    ce_margins = [0.5, 1.0, 1.5, 2.0, 3.0] # Điểm thô CE thường có dải rộng hơn XGBoost
    best_f2 = 0
    best_m = 0

    print("🚀 Đang đánh giá Pure Cross-Encoder (Không dùng XGBoost)...")
    
    for margin in ce_margins:
        f2_list = []
        for item in test_processed:
            # Lấy cột s_ce (index 1)
            s_ce_scores = item["X"].values[:, 1]
            
            # Ghép ID và điểm
            d = {}
            for art, s in zip(item["map"], s_ce_scores):
                if art not in d or s > d[art]: d[art] = s
                
            # Sắp xếp giảm dần theo điểm Cross-Encoder
            art_list = sorted(d.items(), key=lambda x: x[1], reverse=True)
            
            if not art_list:
                f2_list.append(0.0)
                continue
                
            top_score = art_list[0][1]
            preds = [a for a, s in art_list if top_score - s <= margin]
            f2_list.append(calculate_f2(item["gt"], preds))
            
        mean_f2 = np.mean(f2_list)
        if mean_f2 > best_f2:
            best_f2 = mean_f2
            best_m = margin

    print("\n" + "="*50)
    print("🏆 KẾT QUẢ ABLATION STUDY: PURE CROSS-ENCODER 🏆")
    print("="*50)
    print(f"Kiến trúc   : Fine-tuned Cross-Encoder (alqac_model)")
    print(f"Meta-Learner: KHÔNG (Zero-XGBoost)")
    print(f"F2-Macro Đỉnh: {best_f2:.6f} (tại Margin = {best_m})")
    print("="*50)
    
    drop = 0.8772 - best_f2
    print(f"📉 So với việc dùng XGBoost 5 Features, điểm đã giảm: {drop:.4f}")

if __name__ == "__main__":
    run_pure_ce_ablation()