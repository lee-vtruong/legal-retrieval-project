import json
import numpy as np
import pandas as pd

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

def find_best_margin(submission_file):
    with open(submission_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("🔍 Đang rà soát Margin tối ưu cho XGBoost...")
    # Thử nghiệm các khoảng cách margin từ 0.0 (chỉ lấy Top 1) đến 1.0
    margins = np.arange(0.0, 1.0, 0.02)
    best_f2 = 0
    best_m = 0
    
    for m in margins:
        f2_scores = []
        for item in data:
            gt_ids = item.get("relevant_articles", [])
            if not gt_ids: continue
            
            predictions = item.get("predictions", [])
            if not predictions:
                f2_scores.append(0)
                continue
                
            pred_ids = [f"{p['law_id']}_{p['article_id']}" for p in predictions]
            pred_scores = [p["score"] for p in predictions]
            
            # Áp dụng cơ chế Ngưỡng Động
            top_score = pred_scores[0]
            selected_preds = []
            for p_id, score in zip(pred_ids, pred_scores):
                if top_score - score <= m:
                    selected_preds.append(p_id)
                else:
                    break
                    
            f2_scores.append(calculate_f2(gt_ids, selected_preds))
            
        macro_f2 = np.mean(f2_scores)
        if macro_f2 > best_f2:
            best_f2 = macro_f2
            best_m = m

    # Đánh giá lại baseline (Margin = 0.0 tức là Fixed Top-1)
    baseline_f2 = np.mean([calculate_f2(item.get("relevant_articles", []), 
                                        [f"{item['predictions'][0]['law_id']}_{item['predictions'][0]['article_id']}"] 
                                        if item.get("predictions") else []) 
                           for item in data if item.get("relevant_articles")])

    print("="*50)
    print("🎯 KẾT QUẢ TỐI ƯU HÓA NGƯỠNG ĐỘNG XGBOOST")
    print(f"F2 Baseline (Fixed Top-1) : {baseline_f2:.6f}")
    print(f"F2 SOTA (Dynamic Margin)  : {best_f2:.6f} 🌟")
    print(f"Ngưỡng Margin vàng        : {best_m:.2f}")
    print("="*50)

if __name__ == "__main__":
    # Trỏ vào file kết quả của XGBoost mà bạn đã chạy ra lúc nãy
    find_best_margin("data/results/final_submission.json")