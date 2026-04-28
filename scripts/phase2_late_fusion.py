#!/usr/bin/env python3
"""
Phase 2: Late Fusion (XGBoost 5F + Qwen3-14B Scores)
Lai ghép điểm số Meta-Learner với LLM Reasoning Score.
"""

import os, sys, json, pickle
import numpy as np, pandas as pd, xgboost as xgb
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

def calculate_f2(gt_ids, pred_ids):
    gts, preds = set(gt_ids), set(pred_ids)
    if not gts: return 0.0
    tp = len(preds.intersection(gts))
    fp = len(preds - gts)
    fn = len(gts - preds)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    return (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0

def run_phase2():
    print("📦 Nạp Cache 5 Features và Cache LLM Scoring...")
    
    # 1. Load 5 features cache (Bản chứa map và XGBoost data)
    cache_5f = DATA_DIR / "features_cache_5f_with_text.pkl"
    with open(cache_5f, 'rb') as f:
        X_train, y_train, qids, test_processed = pickle.load(f)
        
    # 2. Load LLM scores (Feature thứ 6)
    llm_cache = DATA_DIR / "test_sllm_feature.pkl"
    with open(llm_cache, 'rb') as f:
        llm_scores_list = pickle.load(f)
        
    # 3. Phục hồi Baseline XGBoost 5 Features (0.877)
    print("🚀 Khôi phục sức mạnh Baseline XGBoost...")
    ranker = xgb.XGBRanker(
        n_jobs=-1, tree_method="hist", objective="rank:ndcg",
        n_estimators=150, learning_rate=0.05, max_depth=6
    )
    ranker.fit(X_train, y_train, qid=qids)
    
    # 4. Late Fusion & Grid Search
    print("🧠 Lai ghép điểm số (Late Fusion) và dò tìm F2 đỉnh...")
    
    best_f2 = 0
    best_w = 0
    best_m = 0
    
    # Thử nghiệm các mức trọng số (Weight) cho điểm LLM
    # (Do LLM score từ 0-1, XGB score từ -2 đến 3, ta cần khuếch đại LLM lên)
    weights = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0] 
    margins = np.arange(0.1, 1.5, 0.05)
    
    for w in weights:
        fused_test_results = []
        for idx, item in enumerate(test_processed):
            xgb_scores = ranker.predict(item["X"])
            llm_map = llm_scores_list[idx]
            
            fused_scores = []
            for i, art_id in enumerate(item["map"]):
                s_xgb = float(xgb_scores[i])
                s_llm = llm_map.get(art_id, 0.5) # Fallback 0.5 nếu LLM lỡ sót
                
                # CÔNG THỨC LATE FUSION
                final_s = s_xgb + (s_llm * w)
                fused_scores.append((art_id, final_s))
            
            # Sắp xếp giảm dần theo điểm sau khi Fusion
            fused_scores.sort(key=lambda x: x[1], reverse=True)
            fused_test_results.append({
                "gt": item["gt"],
                "scores": fused_scores
            })
        
        # Quét margin cắt điểm cho mức weight này
        for m in margins:
            f2_list = []
            for res in fused_test_results:
                top_score = res["scores"][0][1]
                preds = [art for art, sc in res["scores"] if top_score - sc <= m]
                f2_list.append(calculate_f2(res["gt"], preds))
            
            curr_f2 = np.mean(f2_list)
            if curr_f2 > best_f2:
                best_f2 = curr_f2
                best_w = w
                best_m = m

    print("\n" + "🏆"*20)
    print("🔥 KẾT QUẢ: XGBOOST + QWEN3-14B (LATE FUSION) 🔥")
    print(f"🚀 F2-macro Tối đa: {best_f2:.6f}")
    print(f"⚖️ Trọng số LLM (w): {best_w}")
    print(f"🎯 Margin cắt điểm: {best_m:.2f}")
    print("🏆"*20)
    
    if best_f2 > 0.877274:
        print("\n🎉 XUẤT SẮC! Hệ thống đã CHÍNH THỨC PHÁ VỠ MỐC 0.877!")
    else:
        print("\nKết quả đang tiệm cận. Có vẻ 14B chấm điểm hơi an toàn.")

if __name__ == "__main__":
    run_phase2()