import json
import os
import sys
import random
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval.hybrid import MultiViewHybridRetriever
from config import TRAIN_FILE, DATA_DIR

def prepare_and_save_data():
    print("🚀 Khởi động Hybrid Retriever...")
    retriever = MultiViewHybridRetriever(load_models=True)
    
    print("📂 Đọc dữ liệu tập Train...")
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # Dictionary tra cứu text
    chunk_dict = {}
    for chunk in retriever.chunks:
        art_key = f"{chunk['law_id']}_{chunk['article_id']}"
        if art_key not in chunk_dict:
            chunk_dict[art_key] = chunk['text']
        else:
            chunk_dict[art_key] += " " + chunk['text']

    export_data = []

    print("🧠 Đang sinh các cặp Positive và Hard Negative...")
    for item in tqdm(train_data):
        query = item.get("text", "")
        gt_ids = [f"{rel['law_id']}_{rel['article_id']}" for rel in item.get("relevant_articles", [])]
        if not gt_ids: continue

        # Positives (Label 1)
        for gt_id in gt_ids:
            if gt_id in chunk_dict:
                export_data.append({"query": query, "passage": chunk_dict[gt_id], "label": 1.0})

        # Hard Negatives (Label 0)
        scores = retriever._get_combined_scores(query)
        top_indices = np.argsort(scores)[::-1][:30]
        
        hn_count = 0
        for idx in top_indices:
            chunk = retriever.chunks[idx]
            art_key = f"{chunk['law_id']}_{chunk['article_id']}"
            
            if art_key not in gt_ids:
                export_data.append({"query": query, "passage": chunk_dict[art_key], "label": 0.0})
                hn_count += 1
            
            if hn_count >= 5: # Tỷ lệ 1 Pos : 5 Neg
                break

    random.shuffle(export_data)
    
    # Lưu xuống ổ cứng
    out_file = DATA_DIR / "ce_training_data.jsonl"
    with open(out_file, 'w', encoding='utf-8') as f:
        for item in export_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Đã lưu xong {len(export_data)} cặp dữ liệu tại: {out_file}")

if __name__ == "__main__":
    prepare_and_save_data()