#!/usr/bin/env python3
"""
Ablation Study Nhóm 2: Đánh giá tầm quan trọng của từng Feature trong XGBoost.
Sử dụng chiến thuật Leave-One-Out trên 5 features gốc.
"""

import os, sys, pickle
import numpy as np, pandas as pd, xgboost as xgb
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

def run_feature_ablation():
    print("📦 Đang nạp file Cache 5 Features (Vui lòng đợi khoảng 10-30 giây)...")
    cache_path = DATA_DIR / "features_cache_5f_with_text.pkl"
    with open(cache_path, 'rb') as f:
        X_train, y_train, qids, test_processed = pickle.load(f)
    print("✅ Đã nạp xong dữ liệu!")

    # Danh sách 5 features gốc
    all_features = ["s_hybrid", "s_ce", "p_ce", "u_uncertainty", "s_gat"]
    MARGIN = 0.7 
    ablation_results = {}

    def train_and_eval(features_to_use, run_name):
        # Lọc cột dữ liệu bằng NumPy array để không bị lỗi Index
        idx_cols = [all_features.index(f) for f in features_to_use]
        X_tr_sub = np.array(X_train)[:, idx_cols]
        
        # GIỚI HẠN n_jobs=8 ĐỂ TRÁNH TREO SERVER DGX
        model = xgb.XGBRanker(n_jobs=8, tree_method="hist", objective="rank:ndcg",
                              n_estimators=150, learning_rate=0.05, max_depth=6)
        
        model.fit(X_tr_sub, y_train, qid=qids)
        
        f2_list = []
        # Thêm tqdm để theo dõi tiến độ chấm điểm
        for item in tqdm(test_processed, desc=f"Đánh giá {run_name}", leave=False):
            X_te_sub = item["X"][features_to_use].values
            f_scores = model.predict(X_te_sub)
            
            d = {}
            for art, s in zip(item["map"], f_scores):
                if art not in d or s > d[art]: d[art] = s
            art_list = sorted(d.items(), key=lambda x: x[1], reverse=True)
            
            top_s = art_list[0][1]
            preds = [a for a, s in art_list if top_s - s <= MARGIN]
            f2_list.append(calculate_f2(item["gt"], preds))
        
        return np.mean(f2_list)

    print("\n🚀 Đang chạy Baseline (Full 5 Features)...")
    baseline_f2 = train_and_eval(all_features, "Baseline")
    ablation_results["Full 5 Features (Baseline)"] = baseline_f2

    # Chạy Leave-One-Out cho từng Feature
    for feat in all_features:
        print(f"\n📉 Đang thử nghiệm: Loại bỏ [{feat}]...")
        sub_features = [f for f in all_features if f != feat]
        f2_val = train_and_eval(sub_features, f"W/o {feat}")
        ablation_results[f"W/o {feat}"] = f2_val

    # In bảng kết quả
    print("\n" + "="*55)
    print("🏆 KẾT QUẢ ABLATION STUDY: FEATURE IMPORTANCE 🏆")
    print("="*55)
    print(f"| {'Cấu hình':<30} | {'F2-Macro':<9} | {'Độ sụt giảm':<11} |")
    print("|" + "-"*32 + "|" + "-"*11 + "|" + "-"*13 + "|")
    
    for name, f2 in ablation_results.items():
        drop = baseline_f2 - f2 if "Full" not in name else 0.0
        # Định dạng dấu + nếu điểm tụt (tức là feature đó quan trọng)
        drop_str = f"{-drop:+.4f}" if drop != 0.0 else "   -   "
        print(f"| {name:<30} | {f2:.4f}    | {drop_str:<11} |")
    print("="*55)

if __name__ == "__main__":
    run_feature_ablation()