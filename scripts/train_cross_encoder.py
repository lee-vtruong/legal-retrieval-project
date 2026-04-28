import json
import os
import sys
import torch
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample

# Import từ Clean Architecture
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, MODELS_DIR, DEVICE, CE_EPOCHS

def train():
    # 1. Đường dẫn dữ liệu và nơi lưu model
    train_samples_path = DATA_DIR / "processed" / "rerank_train_samples.jsonl"
    model_save_path = MODELS_DIR / "reranker_model"
    os.makedirs(model_save_path, exist_ok=True)
    
    print(f"🚀 Nạp dữ liệu huấn luyện từ {train_samples_path}...")
    train_samples = []
    with open(train_samples_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Label 1: Positive, Label 0: Hard Negative
            train_samples.append(InputExample(texts=[data['query'], data['text']], label=float(data['label'])))

    print(f"Tổng số mẫu huấn luyện: {len(train_samples)}")

    # 2. Khởi tạo mô hình Cross-Encoder (Stage 4)
    model_name = "AITeamVN/Vietnamese_Reranker"
    print(f"Khởi tạo mô hình nền: {model_name}...")
    
    # tokenizer_args={'use_fast': False} để tránh lỗi SentencePiece extractor
    model = CrossEncoder(
        model_name, 
        num_labels=1, 
        device=DEVICE, 
        tokenizer_args={'use_fast': False}
    )

    # 3. Thiết lập DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=4)

    # 4. Huấn luyện
    print(f"Bắt đầu Fine-tuning Stage 4 trong {CE_EPOCHS} epochs...")
    num_steps = len(train_dataloader) * CE_EPOCHS
    warmup_steps = int(num_steps * 0.1)

    model.fit(
        train_dataloader=train_dataloader,
        epochs=CE_EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': 2e-5},
        output_path=str(model_save_path),
        save_best_model=False, 
        show_progress_bar=True
    )

    print(f"✅ Đã huấn luyện xong Stage 4 tại: {model_save_path}")

if __name__ == "__main__":
    train()