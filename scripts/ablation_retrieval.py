#!/usr/bin/env python3
"""
Ablation Study Giai đoạn 1: Đánh giá sức mạnh của Multi-View Hybrid Retriever
So sánh Sparse, Dense, Hybrid và Multi-View (GAT) ở Top 15 và Top 30.
"""

import os, sys, json
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval.hybrid import MultiViewHybridRetriever
from config import TEST_FILE

def get_metrics(gt_ids, pred_ids):
    """Tính F2-Macro và Recall"""
    gts, preds = set(gt_ids), set(pred_ids)
    if not gts: return 0.0, 0.0
    
    tp = len(preds & gts)
    fp = len(preds - gts)
    fn = len(gts - preds)
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
    
    return f2, rec

def run_retrieval_ablation():
    print("📚 Khởi tạo Multi-View Hybrid Retriever...")
    retriever = MultiViewHybridRetriever(load_models=True)
    
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        
    print(f"📊 Đang đánh giá {len(test_data)} câu hỏi tập Test...")

    # Dictionary lưu trữ kết quả cho 4 Run ở 2 mức Top-K
    results = {
        "Run 1 (Sparse Only)": {"top15_f2": [], "top15_rec": [], "top30_f2": [], "top30_rec": []},
        "Run 2 (Dense Only)":  {"top15_f2": [], "top15_rec": [], "top30_f2": [], "top30_rec": []},
        "Run 3 (Hybrid)":      {"top15_f2": [], "top15_rec": [], "top30_f2": [], "top30_rec": []},
        "Run 4 (Multi-View)":  {"top15_f2": [], "top15_rec": [], "top30_f2": [], "top30_rec": []}
    }

    for item in tqdm(test_data, desc="Ablation Retrieval"):
        query = item["text"]
        gt_ids = [f"{r['law_id']}_{r['article_id']}" for r in item.get("relevant_articles", [])]
        
        # -----------------------------------------------------------
        # LƯU Ý: Sửa lại tên hàm dưới đây cho khớp với class của bạn
        # -----------------------------------------------------------
        try:
            # 1. Lấy ma trận điểm thô
            sparse_raw = np.array(retriever._get_sparse_scores(query))
            dense_raw = np.array(retriever._get_dense_scores(query))   
            
            # 2. Lấy trung bình điểm của các mô hình cùng loại (ép về mảng 1D)
            sparse_scores = sparse_raw.mean(axis=0) if sparse_raw.ndim > 1 else sparse_raw
            dense_scores = dense_raw.mean(axis=0) if dense_raw.ndim > 1 else dense_raw
            
            # 3. Kết hợp Hybrid (Cộng tuyến tính)
            hybrid_scores = (0.3 * sparse_scores) + (0.7 * dense_scores) 
            
            # 4. Multi-View (Lấy thẳng hàm đã tích hợp GAT của bạn)
            multiview_scores = np.array(retriever._get_combined_scores(query))
            
        except AttributeError:
            print("\n❌ Lỗi: Bạn cần cung cấp/định nghĩa hàm lấy điểm Sparse và Dense trong class MultiViewHybridRetriever.")
            sys.exit(1)

        # Định nghĩa các Run
        runs = {
            "Run 1 (Sparse Only)": sparse_scores,
            "Run 2 (Dense Only)": dense_scores,
            "Run 3 (Hybrid)": hybrid_scores,
            "Run 4 (Multi-View)": multiview_scores
        }

        # Tính toán mốc Top 15 và Top 30
        for run_name, scores in runs.items():
            # Sắp xếp index theo điểm giảm dần
            sorted_idx = np.argsort(scores)[::-1]
            
            # Extract Top 15
            top15_ids = [f"{retriever.chunks[i]['law_id']}_{retriever.chunks[i]['article_id']}" for i in sorted_idx[:15]]
            f2_15, rec_15 = get_metrics(gt_ids, top15_ids)
            results[run_name]["top15_f2"].append(f2_15)
            results[run_name]["top15_rec"].append(rec_15)
            
            # Extract Top 30
            top30_ids = [f"{retriever.chunks[i]['law_id']}_{retriever.chunks[i]['article_id']}" for i in sorted_idx[:30]]
            f2_30, rec_30 = get_metrics(gt_ids, top30_ids)
            results[run_name]["top30_f2"].append(f2_30)
            results[run_name]["top30_rec"].append(rec_30)

    # In kết quả dưới dạng Bảng Markdown để copy thẳng vào Paper
    print("\n" + "="*80)
    print("🏆 KẾT QUẢ ABLATION STUDY: GIAI ĐOẠN RETRIEVAL 🏆")
    print("="*80)
    print(f"| {'Cấu hình (Run)':<22} | {'Top-15 Recall':<13} | {'Top-15 F2':<10} | {'Top-30 Recall':<13} | {'Top-30 F2':<10} |")
    print("|" + "-"*24 + "|" + "-"*15 + "|" + "-"*12 + "|" + "-"*15 + "|" + "-"*12 + "|")
    
    for run_name in results.keys():
        rec15 = np.mean(results[run_name]["top15_rec"])
        f2_15 = np.mean(results[run_name]["top15_f2"])
        rec30 = np.mean(results[run_name]["top30_rec"])
        f2_30 = np.mean(results[run_name]["top30_f2"])
        
        print(f"| {run_name:<22} | {rec15:.4f}        | {f2_15:.4f}     | {rec30:.4f}        | {f2_30:.4f}     |")
    print("="*80)

if __name__ == "__main__":
    run_retrieval_ablation()