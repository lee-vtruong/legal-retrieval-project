import json
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Import từ hệ thống Clean Architecture
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval.hybrid import MultiViewHybridRetriever
from config import TEST_FILE, DATA_DIR

def calculate_all_metrics(gt_ids, pred_ids):
    """Tính toán bộ metrics hoàn hảo cho một câu hỏi"""
    preds_10 = pred_ids[:10]
    gts = set(gt_ids)
    R = len(gts)
    
    # Nếu không có Ground Truth thì không thể tính metrics
    if R == 0:
        return None
        
    # --- Metrics tại K=10 ---
    preds_set = set(preds_10)
    tp = len(preds_set.intersection(gts))
    fp = len(preds_set - gts)
    fn = len(gts - preds_set)
    
    prec_10 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_10 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f2_10 = (5 * prec_10 * recall_10) / (4 * prec_10 + recall_10) if (4 * prec_10 + recall_10) > 0 else 0
    
    # --- mAP (Mean Average Precision) ---
    avg_prec = 0
    hits = 0
    for i, p_id in enumerate(preds_10):
        if p_id in gts:
            hits += 1
            avg_prec += hits / (i + 1)
    map_score = avg_prec / R if R > 0 else 0
    
    # --- R-Precision ---
    r_preds = set(pred_ids[:R])
    r_hits = len(r_preds.intersection(gts))
    r_prec = r_hits / R if R > 0 else 0
    
    # --- MRR (Mean Reciprocal Rank) ---
    mrr = 0
    for i, p_id in enumerate(pred_ids):
        if p_id in gts:
            mrr = 1 / (i + 1)
            break
            
    return {
        "F2-macro": f2_10,
        "mAP": map_score,
        "R-Prec": r_prec,
        "MRR": mrr,
        "Recall@1": 1 if any(p in gts for p in pred_ids[:1]) else 0,
        "Recall@5": 1 if any(p in gts for p in pred_ids[:5]) else 0,
        "Recall@10": 1 if tp > 0 else 0
    }

def run_ultimate_evaluation():
    print(f"🚀 Nạp mô hình Multi-View Hybrid Retriever...")
    retriever = MultiViewHybridRetriever(load_models=True)
    
    print(f"📂 Đọc dữ liệu đánh giá từ: {TEST_FILE}")
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # Thư mục lưu kết quả
    run_id = datetime.now().strftime('%m%d_%H%M')
    results_dir = DATA_DIR / "results" / f"ablation_s1_{run_id}"
    os.makedirs(results_dir, exist_ok=True)

    # Lọc index cho Baseline (Chỉ lấy Article)
    article_only_indices = [
        i for i, chunk in enumerate(retriever.chunks) 
        if "_para_" not in chunk['chunk_id'] and "_sent_" not in chunk['chunk_id']
    ]

    test_configs = {
        "S1_Baseline_Article": True,
        "S1_Proposed_Multi": False
    }

    final_summary = []

    for name, is_baseline in test_configs.items():
        print(f"\n🔥 Đang đánh giá: {name} trên tập Test...")
        detailed_log = []
        metrics_accumulator = []

        for item in tqdm(test_data):
            query = item.get("text", "")
            gt_ids = [f"{rel['law_id']}_{rel['article_id']}" for rel in item.get("relevant_articles", [])]
            
            if not gt_ids:
                continue
            
            # 1. Lấy điểm và Gom nhóm theo Article (Max Aggregation)
            scores = retriever._get_combined_scores(query)
            article_scores = {}
            for idx, score in enumerate(scores):
                if is_baseline and idx not in article_only_indices:
                    continue
                chunk = retriever.chunks[idx]
                key = f"{chunk['law_id']}_{chunk['article_id']}"
                if key not in article_scores or score > article_scores[key]:
                    article_scores[key] = score
            
            # Xếp hạng Điều luật
            sorted_arts = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)
            pred_ids = [art[0] for art in sorted_arts]
            
            # 2. Tính toán bộ metrics "hoàn hảo"
            q_metrics = calculate_all_metrics(gt_ids, pred_ids)
            if q_metrics is not None:
                metrics_accumulator.append(q_metrics)

                # 3. Lưu log chi tiết
                detailed_log.append({
                    "query": query,
                    "ground_truth": gt_ids,
                    "top_10_preds": pred_ids[:10],
                    "metrics": q_metrics
                })

        # Lưu file JSON chi tiết
        with open(results_dir / f"{name}_details.json", "w", encoding="utf-8") as f:
            json.dump(detailed_log, f, ensure_ascii=False, indent=4)

        # Tính trung bình cộng
        if metrics_accumulator:
            avg_res = pd.DataFrame(metrics_accumulator).mean().to_dict()
            avg_res["Configuration"] = name
            final_summary.append(avg_res)

    # Xuất CSV tổng hợp
    if final_summary:
        summary_df = pd.DataFrame(final_summary)
        # Tính ∆ F2 so với baseline
        base_f2 = summary_df.iloc[0]["F2-macro"]
        summary_df["∆_F2"] = summary_df["F2-macro"] - base_f2
        
        # Đưa cột Configuration lên đầu để dễ nhìn
        cols = ['Configuration', 'F2-macro', '∆_F2', 'mAP', 'R-Prec', 'Recall@10', 'MRR']
        # Đảm bảo các cột có tồn tại trước khi select
        cols = [c for c in cols if c in summary_df.columns] 
        summary_df = summary_df[cols]

        summary_df.to_csv(results_dir / "full_metrics_summary.csv", index=False)
        
        print("\n" + "="*80)
        print(f"KẾT QUẢ THỰC NGHIỆM STAGE 1 TRÊN TẬP TEST (Lưu tại: {results_dir})")
        print(summary_df.to_string(index=False))
        print("="*80)
    else:
        print("❌ Không có dữ liệu hợp lệ để tính toán metrics.")

if __name__ == "__main__":
    run_ultimate_evaluation()