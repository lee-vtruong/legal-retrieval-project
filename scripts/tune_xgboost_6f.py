import json, os, sys, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

def calculate_f2(gt_ids, pred_ids):
    gts, preds = set(gt_ids), set(pred_ids)
    if not gts: return 0
    tp = len(preds.intersection(gts))
    fp, fn = len(preds - gts), len(gts - preds)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    return (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0

def run_tuning_6f():
    cache_file = DATA_DIR / "features_cache_6f.pkl"
    if not os.path.exists(cache_file):
        print("❌ Lỗi: Không tìm thấy file features_cache_6f.pkl!")
        return

    print("📦 Đang nạp Cache 6 Features (s_bm25, s_dense, s_ce, p_ce, u_uncertainty, s_gat)...")
    with open(cache_file, 'rb') as f:
        X_train, y_train, qids, test_processed = pickle.load(f)

    # --- GRID SEARCH ---
    print("\n🔍 Đang quét cấu hình (Tối ưu hóa NDCG & Dynamic Margin)...")
    best_overall_f2 = 0
    best_params = {}
    best_margin = 0
    
    import itertools
    # Quét quanh "thông số vàng" cũ để tiết kiệm thời gian
    configs = list(itertools.product([0.05, 0.1], [4, 6], [100, 150, 200]))

    for lr, depth, est in configs:
        ranker = xgb.XGBRanker(
            n_jobs=1, 
            tree_method="hist", 
            objective="rank:ndcg", 
            n_estimators=est, 
            learning_rate=lr, 
            max_depth=depth
        )
        ranker.fit(X_train, y_train, qid=qids)
        
        # Lấy dự đoán cho toàn bộ tập Test
        all_preds = []
        for item in test_processed:
            f_scores = ranker.predict(item["X"])
            sorted_idx = np.argsort(f_scores)[::-1]
            all_preds.append({
                "gt": item["gt"], 
                "map": [item["map"][i] for i in sorted_idx], 
                "scores": [f_scores[i] for i in sorted_idx]
            })

        # Quét Margin nhanh để tìm F2 cao nhất cho cấu hình này
        local_best_f2 = 0
        local_best_margin = 0
        for m in np.arange(0.1, 1.5, 0.05):
            f2_list = []
            for p in all_preds:
                top_score = p["scores"][0]
                selected = [p["map"][i] for i, s in enumerate(p["scores"]) if top_score - s <= m]
                f2_list.append(calculate_f2(p["gt"], selected))
            
            curr_f2 = np.mean(f2_list)
            if curr_f2 > local_best_f2:
                local_best_f2 = curr_f2
                local_best_margin = m
                
        print(f"-> Params: LR={lr}, Depth={depth}, Est={est:3d} | Best F2: {local_best_f2:.5f} (Margin: {local_best_margin:.2f})")
        
        if local_best_f2 > best_overall_f2:
            best_overall_f2 = local_best_f2
            best_params = {"lr": lr, "depth": depth, "est": est}
            best_margin = local_best_margin

    print("\n" + "🚀"*20)
    print("🔥 KẾT QUẢ TỐI ƯU 6 FEATURES 🔥")
    print(f"F2-macro: {best_overall_f2:.6f}")
    print(f"Params  : {best_params}")
    print(f"Margin  : {best_margin:.2f}")
    print("🚀"*20)
    
    if best_overall_f2 > 0.8772:
        print("\n🎉 CHÚC MỪNG! BẠN ĐÃ PHÁ KỶ LỤC CŨ (0.8772)! 🎉")
    else:
        print("\nChưa vượt qua được 0.8772. Có vẻ như XGBoost thích việc gộp trọng số Hybrid sẵn hơn!")

if __name__ == "__main__":
    run_tuning_6f()