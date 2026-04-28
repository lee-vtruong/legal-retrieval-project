#!/usr/bin/env python3
"""
Ablation Study Nhóm 4 (Run 11): Early Fusion
Đưa điểm LLM (s_llm) vào làm Feature thứ 6 để XGBoost tự học.
"""

import os, sys, pickle, json
import numpy as np, pandas as pd, xgboost as xgb
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

def calculate_f2(gt_ids, pred_ids):
    gts, preds = set(gt_ids), set(pred_ids)
    if not gts: return 0.0
    tp = len(preds & gts)
    fp, fn = len(preds - gts), len(gts - preds)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    return (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0

def run_early_fusion():
    # 1. Nạp cache 5 features (Train & Test)
    print("📦 Đang nạp Cache 5 Features...")
    cache_path = DATA_DIR / "features_cache_5f_with_text.pkl"
    with open(cache_path, 'rb') as f:
        X_train_5f, y_train, qids, test_processed = pickle.load(f)

    # 2. Nạp điểm LLM (Feature thứ 6 cho tập Test)
    print("🤖 Đang nạp điểm LLM (Feature thứ 6)...")
    llm_cache = DATA_DIR / "test_sllm_feature.pkl"
    with open(llm_cache, 'rb') as f:
        llm_scores_list = pickle.load(f)

    # 3. Chuẩn bị dữ liệu Train (6 Features)
    # Vì chưa có điểm LLM cho Train, ta để giá trị trung bình 0.5
    print("🛠️ Đang chuẩn bị ma trận 6-Features...")
    X_train_6f = np.hstack([np.array(X_train_5f), np.full((X_train_5f.shape[0], 1), 0.5)])

    # 4. Huấn luyện XGBoost 6-Features
    print("🚀 Đang huấn luyện XGBRanker với 6 Features (Early Fusion)...")
    model = xgb.XGBRanker(
        n_jobs=8, tree_method="hist", objective="rank:ndcg",
        n_estimators=150, learning_rate=0.05, max_depth=6
    )
    model.fit(X_train_6f, y_train, qid=qids)

    # 5. Đánh giá trên tập Test
    f2_list = []
    MARGIN = 0.7 # Dùng lại margin chuẩn của baseline để so sánh công bằng

    for idx, item in enumerate(tqdm(test_processed, desc="Evaluating Early Fusion")):
        # Lấy điểm LLM cho query hiện tại
        llm_map = llm_scores_list[idx]
        
        # Tạo cột thứ 6 cho dữ liệu test hiện tại
        s_llm_col = np.array([llm_map.get(art_id, 0.5) for art_id in item["map"]]).reshape(-1, 1)
        X_test_6f = np.hstack([item["X"].values, s_llm_col])
        
        # Dự đoán
        f_scores = model.predict(X_test_6f)
        
        # Chốt đáp án với Dynamic Margin
        d = {art: s for art, s in zip(item["map"], f_scores)}
        art_list = sorted(d.items(), key=lambda x: x[1], reverse=True)
        
        top_s = art_list[0][1]
        preds = [a for a, s in art_list if top_s - s <= MARGIN]
        f2_list.append(calculate_f2(item["gt"], preds))

    final_f2 = np.mean(f2_list)
    print("\n" + "="*50)
    print("🏆 KẾT QUẢ ABLATION: EARLY FUSION (6-FEATURES) 🏆")
    print("="*50)
    print(f"Kiến trúc   : XGBoost Meta-Learner (NDCG)")
    print(f"Số lượng Feature: 6 (Thêm s_llm)")
    print(f"F2-Macro Đỉnh: {final_f2:.6f}")
    print("="*50)
    
    print(f"\n📊 So sánh:")
    print(f"- Baseline (5F)    : 0.8773")
    print(f"- Late Fusion (5F+LLM): 0.8996")
    print(f"- Early Fusion (6F)   : {final_f2:.6f}")

if __name__ == "__main__":
    run_early_fusion()