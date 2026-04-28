import json
import os
import sys
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from sentence_transformers import CrossEncoder

# Import từ Clean Architecture
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval.hybrid import MultiViewHybridRetriever
from config import DATA_DIR, MODELS_DIR, DEVICE, TEST_FILE

# LƯU Ý: Đổi đường dẫn này thành tập Test/Val có chứa label của bạn
# Giả sử chúng ta đang chạy lại trên tập val hoặc test_gold để lấy metrics
# TEST_FILE = DATA_DIR / "raw" / "private_test_GOLD.json" # <--- Sửa nếu cần

def calculate_all_metrics(gt_ids, pred_ids, scores=None):
    """Tính toán bộ metrics chuẩn ALQAC"""
    gts = set(gt_ids)
    R = len(gts)
    if R == 0: return None
    
    # --- SỬA LỖI Ở ĐÂY: Dùng Top 1 thay vì Top 10 để tính F2 ---
    # Vì đa số câu hỏi ALQAC có 1 đáp án, Top 1 sẽ tối ưu F2 nhất.
    # Nếu hệ thống có Threshold, bạn có thể lọc động dựa vào `scores`.
    preds_for_f2 = set(pred_ids[:1]) # Chỉ lấy Top 1 để nộp
    
    tp = len(preds_for_f2.intersection(gts))
    fp = len(preds_for_f2 - gts)
    fn = len(gts - preds_for_f2)
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f2_macro = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0
    
    # --- Các metrics xếp hạng (Giữ nguyên vì nó đánh giá Ranking) ---
    preds_10 = pred_ids[:10]
    avg_prec = 0
    hits = 0
    for i, p_id in enumerate(preds_10):
        if p_id in gts:
            hits += 1
            avg_prec += hits / (i + 1)
    map_score = avg_prec / R if R > 0 else 0
    
    r_preds = set(pred_ids[:R])
    r_hits = len(r_preds.intersection(gts))
    r_prec = r_hits / R if R > 0 else 0
    
    mrr = 0
    for i, p_id in enumerate(pred_ids):
        if p_id in gts:
            mrr = 1 / (i + 1)
            break
            
    return {
        "F2-macro": f2_macro, 
        "mAP": map_score,
        "R-Prec": r_prec,
        "Recall@10": 1 if any(p in gts for p in preds_10) else 0,
        "MRR": mrr
    }

def run_inference():
    print("🔥 KHỞI ĐỘNG HỆ THỐNG GAM-HYBRID (FULL PIPELINE) 🔥")
    
    # 1. Load các thành phần (Chạy trên GPU 5)
    print("1. Nạp Multi-Granular Hybrid & GAT Retriever...")
    retriever = MultiViewHybridRetriever(load_models=True)
    
    print("2. Nạp Cross-Encoder...")
    ce_model = CrossEncoder(str(MODELS_DIR / "reranker_model"), device="cuda:0")
    
    print("3. Nạp Meta-Learner (XGBRanker)...")
    ranker = xgb.XGBRanker()
    ranker.load_model(str(MODELS_DIR / "xgboost_ranker.json"))
    features_order = ["s_hybrid", "s_ce", "p_ce", "u_uncertainty", "s_gat"]

    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    results_log = []
    metrics_list = []
    TOP_K_RETRIEVAL = 30 # Đưa 30 ứng viên vào cho Meta-Learner chốt

    print(f"\n🚀 Bắt đầu suy luận trên {len(test_data)} câu hỏi...")
    for item in tqdm(test_data):
        query = item.get("text", "")
        gt_articles = [f"{rel['law_id']}_{rel['article_id']}" for rel in item.get("relevant_articles", [])]
        
        # --- BƯỚC 1: RETRIEVAL (STAGE 2 & 3) ---
        scores = retriever._get_combined_scores(query)
        top_indices = np.argsort(scores)[::-1][:TOP_K_RETRIEVAL]
        
        # --- BƯỚC 2: CROSS-ENCODER (STAGE 4) ---
        pairs = [[query, retriever.chunks[idx]['text']] for idx in top_indices]
        ce_raw_scores = ce_model.predict(pairs, batch_size=32, convert_to_numpy=True)
        ce_probs = torch.nn.functional.softmax(torch.tensor(ce_raw_scores), dim=0).numpy()
        
        sorted_probs = np.sort(ce_probs)[::-1]
        u_val = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0

        # --- BƯỚC 3: META-LEARNER FUSION (STAGE 5) ---
        gat_norm = retriever._get_graph_scores(query)
        
        df_features = []
        mapping_idx = [] # Ánh xạ lại index của DataFrame về chunk index
        
        for i, idx in enumerate(top_indices):
            chunk = retriever.chunks[idx]
            art_key = f"{chunk['law_id']}_{chunk['article_id']}"
            gat_score = gat_norm[retriever.gat_mapping[art_key]] if art_key in retriever.gat_mapping else 0.0
            
            df_features.append({
                "s_hybrid": float(scores[idx]),
                "s_ce": float(ce_raw_scores[i]),
                "p_ce": float(ce_probs[i]),
                "u_uncertainty": float(u_val),
                "s_gat": float(gat_score)
            })
            mapping_idx.append((idx, art_key))

        X_test = pd.DataFrame(df_features)[features_order]
        
        # XGBRanker chấm điểm cuối cùng
        final_scores = ranker.predict(X_test)
        
        # --- BƯỚC 4: GOM NHÓM ĐIỀU LUẬT (AGGREGATION) ---
        article_final_scores = {}
        for i, f_score in enumerate(final_scores):
            idx, art_key = mapping_idx[i]
            if art_key not in article_final_scores or f_score > article_final_scores[art_key]:
                article_final_scores[art_key] = f_score
                
        sorted_arts = sorted(article_final_scores.items(), key=lambda x: x[1], reverse=True)
        pred_ids = [art[0] for art in sorted_arts]
        pred_scores = [float(art[1]) for art in sorted_arts]

        # Tính metrics nếu có ground truth
        if gt_articles:
            q_metrics = calculate_all_metrics(gt_articles, pred_ids)
            if q_metrics:
                metrics_list.append(q_metrics)
        else:
            q_metrics = {}

        # Lưu kết quả
        results_log.append({
            "question_id": item.get("question_id", "N/A"),
            "text": query,
            "relevant_articles": gt_articles,
            "predictions": [
                {"law_id": p.split("_")[0], "article_id": p.split("_")[1], "score": round(s, 4)} 
                for p, s in zip(pred_ids, pred_scores)
            ]
        })

    # --- BƯỚC 5: XUẤT BÁO CÁO ---
    os.makedirs(DATA_DIR / "results", exist_ok=True)
    
    # 1. Lưu file JSON
    out_json = DATA_DIR / "results" / "final_submission.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=4)
        
    # 2. In Metrics
    if metrics_list:
        summary_df = pd.DataFrame(metrics_list).mean().to_frame().T
        summary_df["Configuration"] = "GAM-Hybrid (Full System)"
        
        out_csv = DATA_DIR / "results" / "final_metrics.csv"
        summary_df.to_csv(out_csv, index=False)
        
        print("\n" + "🌟"*25)
        print("KẾT QUẢ ĐÁNH GIÁ HỆ THỐNG GAM-HYBRID (STAGE 5)")
        print(summary_df[["Configuration", "F2-macro", "mAP", "R-Prec", "Recall@10", "MRR"]].to_string(index=False))
        print("🌟"*25)
        print(f"✅ Đã lưu Predictions tại: {out_json}")
    else:
        print(f"\n✅ Đã chạy xong! Không có Ground Truth để tính metrics. Output lưu tại: {out_json}")

if __name__ == "__main__":
    run_inference()