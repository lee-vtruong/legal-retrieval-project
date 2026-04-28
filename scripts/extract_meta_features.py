import json
import os
import sys
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import CrossEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_FILE, DATA_DIR, MODELS_DIR, DEVICE
from src.retrieval.hybrid import MultiViewHybridRetriever

def extract():
    # 1. Khởi tạo
    print("Khởi tạo Hybrid Retriever...")
    retriever = MultiViewHybridRetriever(load_models=True)
    
    ce_model_path = MODELS_DIR / "reranker_model"
    print(f"Khởi tạo Cross-Encoder từ {ce_model_path}...")
    ce_model = CrossEncoder(str(ce_model_path), device=DEVICE)

    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    meta_rows = []
    TOP_K = 30 # Đào top 30 ứng viên để XGBRanker học

    print("🚀 Bắt đầu trích xuất Meta-features...")
    for q_idx, item in enumerate(tqdm(train_data)):
        query = item.get("text", "")
        gt_articles = [f"{rel['law_id']}_{rel['article_id']}" for rel in item.get("relevant_articles", [])]
        if not gt_articles: continue
        
        # --- STAGE 2 & 3: Lấy điểm Hybrid + GAT ---
        scores = retriever._get_combined_scores(query)
        
        # Lấy Top-K index
        top_indices = np.argsort(scores)[::-1][:TOP_K]
        
        # Chuẩn bị cặp câu cho Cross-Encoder
        pairs = []
        for idx in top_indices:
            chunk = retriever.chunks[idx]
            pairs.append([query, chunk['text']])
            
        # --- STAGE 4: Chấm điểm Cross-Encoder ---
        ce_raw_scores = ce_model.predict(pairs, batch_size=32, convert_to_numpy=True)
        ce_probs = torch.nn.functional.softmax(torch.tensor(ce_raw_scores), dim=0).numpy()
        
        # Tính Uncertainty (U) = Xác suất lớn nhất - Xác suất lớn thứ hai
        sorted_probs = np.sort(ce_probs)[::-1]
        u_val = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0

        # --- TỔNG HỢP ĐẶC TRƯNG ---
        gat_norm = retriever._get_graph_scores(query)

        for i, idx in enumerate(top_indices):
            chunk = retriever.chunks[idx]
            art_key = f"{chunk['law_id']}_{chunk['article_id']}"
            is_positive = 1 if art_key in gt_articles else 0
            
            gat_score = gat_norm[retriever.gat_mapping[art_key]] if art_key in retriever.gat_mapping else 0.0

            meta_rows.append({
                "qid": q_idx, 
                "s_hybrid": float(scores[idx]), # Điểm của Stage 2+3
                "s_ce": float(ce_raw_scores[i]), # Raw score Stage 4
                "p_ce": float(ce_probs[i]),      # Xác suất
                "u_uncertainty": float(u_val),   # Độ bất định
                "s_gat": float(gat_score),       # Tín hiệu đồ thị riêng
                "label": is_positive
            })

    # 2. Lưu file CSV
    df = pd.DataFrame(meta_rows)
    output_dir = DATA_DIR / "processed"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = output_dir / "meta_train_features.csv"
    df.to_csv(csv_path, index=False)
    
    # 3. Tạo file group cho XGBoost
    group_counts = df.groupby("qid").size().tolist()
    group_path = output_dir / "meta_train_groups.txt"
    with open(group_path, "w") as f:
        for count in group_counts:
            f.write(f"{count}\n")

    print(f"✅ Xong! Đặc trưng lưu tại: {csv_path}")

if __name__ == "__main__":
    extract()