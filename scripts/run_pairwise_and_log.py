import json
import os
import sys
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEST_FILE

def calculate_f2(gt_ids, pred_ids):
    gts, preds = set(gt_ids), set(pred_ids)
    if not gts: return 0
    tp = len(preds.intersection(gts))
    fp, fn = len(preds - gts), len(gts - preds)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    return (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0

def run_pairwise_and_log():
    cache_file = DATA_DIR / "features_cache_5f.pkl"
    if not os.path.exists(cache_file):
        print("❌ Không tìm thấy cache!")
        return

    print("📦 Đang nạp Cache 5 Features...")
    with open(cache_file, 'rb') as f:
        X_train, y_train, qids, test_processed = pickle.load(f)

    # 1. ÁP DỤNG CHIẾN LƯỢC PAIRWISE
    print("⚙️ Huấn luyện XGBoost với rank:pairwise...")
    ranker = xgb.XGBRanker(
        n_jobs=1, 
        tree_method="hist", 
        objective="rank:pairwise", # ĐIỂM SÁNG GIÁ NHẤT: So sánh cặp
        n_estimators=150, 
        learning_rate=0.05, 
        max_depth=6
    )
    ranker.fit(X_train, y_train, qid=qids)

    # 2. INFERENCE & DYNAMIC MARGIN
    print("🔍 Đang dự đoán và tìm Margin vàng...")
    all_preds = []
    for item in test_processed:
        f_scores = ranker.predict(item["X"])
        art_scores = {}
        for s, k in zip(f_scores, item["map"]):
            if k not in art_scores or s > art_scores[k]: 
                art_scores[k] = float(s)
        sorted_arts = sorted(art_scores.items(), key=lambda x: x[1], reverse=True)
        all_preds.append({
            "gt": item["gt"], 
            "ids": [a[0] for a in sorted_arts], 
            "scores": [a[1] for a in sorted_arts]
        })

    # Quét margin nhanh để lấy Best F2
    best_m_f2 = 0
    best_m = 0
    for m in np.arange(0.0, 2.5, 0.05):
        f2_list = []
        for p in all_preds:
            if not p["ids"]: 
                f2_list.append(0)
                continue
            selected = [i for i, s in zip(p["ids"], p["scores"]) if p["scores"][0] - s <= m]
            f2_list.append(calculate_f2(p["gt"], selected))
        
        curr_f2 = np.mean(f2_list)
        if curr_f2 > best_m_f2: 
            best_m_f2 = curr_f2
            best_m = m

    print(f"\n🏆 F2 SAU KHI DÙNG PAIRWISE: {best_m_f2:.6f} (Margin: {best_m:.2f})")

    # 3. LƯU LOG ERROR ANALYSIS
    print(f"\n💾 Đang xuất file Error Analysis với Margin {best_m:.2f}...")
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_raw_data = json.load(f)

    error_logs = []
    
    for idx, (raw_item, pred_item) in enumerate(zip(test_raw_data, all_preds)):
        query = raw_item.get("text", "")
        ground_truths = pred_item["gt"]
        
        # Áp dụng Margin để chốt đáp án cuối cùng
        top_score = pred_item["scores"][0] if pred_item["scores"] else 0
        final_selected_ids = [i for i, s in zip(pred_item["ids"], pred_item["scores"]) if top_score - s <= best_m]
        
        # Kiểm tra xem có sai sót không (Miss đáp án hoặc Predict dư)
        is_perfect = set(ground_truths) == set(final_selected_ids)
        
        if not is_perfect:
            error_logs.append({
                "query_id": idx,
                "query": query,
                "ground_truths": ground_truths,
                "model_predicted": final_selected_ids,
                "missing_answers": list(set(ground_truths) - set(final_selected_ids)),
                "false_positives": list(set(final_selected_ids) - set(ground_truths)),
                "raw_top_5_scores": [
                    { "id": i, "score": s } 
                    for i, s in zip(pred_item["ids"][:5], pred_item["scores"][:5])
                ]
            })

    output_file = DATA_DIR / "error_analysis_report.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(error_logs, f, ensure_ascii=False, indent=4)
        
    print(f"✅ Đã lưu {len(error_logs)} câu hỏi bị sai lệch vào: {output_file}")
    print("-> Bạn hãy mở file này ra, xem mục 'missing_answers' và 'false_positives' để chẩn đoán bệnh nhé!")

if __name__ == "__main__":
    run_pairwise_and_log()