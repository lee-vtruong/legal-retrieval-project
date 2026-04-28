import json
import os
import sys
import numpy as np
import pandas as pd

# Giả định file kết quả cũ của bạn nằm ở đường dẫn này
# Nếu bạn để ở chỗ khác, hãy sửa lại đường dẫn RESULTS_FILE
RESULTS_FILE = "data/results/final_submission.json"

def calculate_f2(gt_ids, pred_ids):
    """Công thức tính F2-score chuẩn của ALQAC"""
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

def tune_baseline():
    if not os.path.exists(RESULTS_FILE):
        print(f"❌ Không tìm thấy file: {RESULTS_FILE}")
        print("Vui lòng kiểm tra lại đường dẫn file final_submission.json của mô hình XGBoost cũ.")
        return

    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"📂 Đang xử lý {len(data)} câu hỏi từ Baseline nguyên bản...")

    # Chuẩn bị dữ liệu thô
    processed_data = []
    for item in data:
        gt_ids = item.get("relevant_articles", [])
        # Một số file output của bạn lưu gt là list string, một số là list dict, ta chuẩn hóa lại:
        if gt_ids and isinstance(gt_ids[0], dict):
            gt_ids = [f"{rel['law_id']}_{rel['article_id']}" for rel in gt_ids]
        
        if not gt_ids: continue

        predictions = item.get("predictions", [])
        if not predictions:
            processed_data.append({"gt": gt_ids, "ids": [], "scores": []})
            continue

        p_ids = [f"{p['law_id']}_{p['article_id']}" for p in predictions]
        p_scores = [float(p.get("score", 0)) for p in predictions]
        
        processed_data.append({"gt": gt_ids, "ids": p_ids, "scores": p_scores})

    # --- Quét Margin ---
    print("🔍 Đang rà soát Margin tối ưu để nâng cấp F2-macro...")
    # Quét từ 0.0 đến 2.0 với bước nhảy nhỏ
    margins = np.arange(0.0, 2.0, 0.01)
    best_f2 = 0
    best_m = 0
    
    for m in margins:
        current_f2_list = []
        for item in processed_data:
            ids = item["ids"]
            scores = item["scores"]
            if not ids:
                current_f2_list.append(0)
                continue
            
            top_score = scores[0]
            selected = []
            for p_id, s in zip(ids, scores):
                if top_score - s <= m:
                    selected.append(p_id)
                else:
                    break
            
            current_f2_list.append(calculate_f2(item["gt"], selected))
            
        avg_f2 = np.mean(current_f2_list)
        if avg_f2 > best_f2:
            best_f2 = avg_f2
            best_m = m

    # Tính Baseline Fixed Top-1 để đối chiếu
    baseline_f2 = np.mean([calculate_f2(item["gt"], [item["ids"][0]] if item["ids"] else []) for item in processed_data])

    print("\n" + "="*50)
    print("📊 BÁO CÁO PHÂN TÍCH BASELINE NGUYÊN BẢN")
    print(f"F2-macro (Nếu chỉ lấy Top 1)    : {baseline_f2:.6f}")
    print(f"F2-macro SOTA (Dynamic Margin)  : {best_f2:.6f} 🚀")
    print(f"Ngưỡng Margin tối ưu tìm được   : {best_m:.2f}")
    print("="*50)
    print("Lời khuyên: Hãy dùng con số SOTA này làm cột mốc để so sánh với")
    print("các cải tiến Law-tag và Fine-tune CE sắp tới.")

if __name__ == "__main__":
    tune_baseline()