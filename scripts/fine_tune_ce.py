import json
import os
import sys
import random
import numpy as np
from tqdm import tqdm
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval.hybrid import MultiViewHybridRetriever
from config import TRAIN_FILE, MODELS_DIR

def prepare_training_data(retriever):
    print("📂 Đọc dữ liệu tập Train...")
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    train_examples = []
    
    # Gom tất cả các chunk text vào một dictionary để tra cứu nhanh theo ID
    chunk_dict = {}
    for chunk in retriever.chunks:
        art_key = f"{chunk['law_id']}_{chunk['article_id']}"
        if art_key not in chunk_dict:
            chunk_dict[art_key] = chunk['text']
        else:
            # Nếu article có nhiều chunk, ghép lại hoặc lấy chunk đầu
            chunk_dict[art_key] += " " + chunk['text']

    print("🧠 Đang sinh các cặp Positive và Hard Negative...")
    for item in tqdm(train_data):
        query = item.get("text", "")
        gt_ids = [f"{rel['law_id']}_{rel['article_id']}" for rel in item.get("relevant_articles", [])]
        if not gt_ids: continue

        # 1. Thêm các cặp Positive (Nhãn 1.0)
        for gt_id in gt_ids:
            if gt_id in chunk_dict:
                train_examples.append(InputExample(texts=[query, chunk_dict[gt_id]], label=1.0))

        # 2. Dùng Hybrid Retriever để tìm Hard Negatives
        scores = retriever._get_combined_scores(query)
        top_indices = np.argsort(scores)[::-1][:30] # Lấy Top 30 làm bẫy
        
        hard_negatives_added = 0
        for idx in top_indices:
            chunk = retriever.chunks[idx]
            art_key = f"{chunk['law_id']}_{chunk['article_id']}"
            
            # Nếu nó nằm trong Top 30 nhưng KHÔNG PHẢI là đáp án đúng -> Nó là Hard Negative
            if art_key not in gt_ids:
                train_examples.append(InputExample(texts=[query, chunk_dict[art_key]], label=0.0))
                hard_negatives_added += 1
            
            # Chỉ lấy khoảng 5-7 Hard Negatives cho mỗi câu hỏi để cân bằng dữ liệu (Tỷ lệ Pos/Neg ~ 1:5)
            if hard_negatives_added >= 5:
                break

    random.shuffle(train_examples)
    print(f"✅ Đã tạo xong {len(train_examples)} cặp huấn luyện (Positives + Hard Negatives).")
    return train_examples

def run_fine_tuning():
    print("🚀 Khởi động Hybrid Retriever để đào dữ liệu...")
    retriever = MultiViewHybridRetriever(load_models=True)
    
    train_examples = prepare_training_data(retriever)
    
    # Batch size lớn tận dụng VRAM của A100
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

    print("\n🔥 Nạp pre-trained Cross-Encoder hiện tại...")
    # Nạp lại mô hình cũ của bạn để train tiếp (hoặc dùng vinai/phobert-base-v2 nếu muốn train từ đầu)
    model_path = str(MODELS_DIR / "reranker_model") 
    model = CrossEncoder(model_path, num_labels=1, device="cuda:0")

    # Đường dẫn lưu model mới
    output_path = str(MODELS_DIR / "fine_tuned_ce_alqac")

    print(f"⚙️ Bắt đầu Fine-tuning (1-2 Epochs là đủ tránh Overfitting)...")
    # Cấu hình hàm Loss mặc định của CrossEncoder cho label float (0.0 và 1.0) là BCEWithLogitsLoss
    model.fit(
        train_dataloader=train_dataloader,
        epochs=2,
        warmup_steps=100,
        output_path=output_path,
        show_progress_bar=True
    )
    
    print("\n" + "🌟"*25)
    print(f"✅ Đã Fine-tune xong! Mô hình CE mới lưu tại: {output_path}")
    print("🌟"*25)

if __name__ == "__main__":
    run_fine_tuning()