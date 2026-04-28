import json
import os
import sys
import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, DATA_DIR

def run_fine_tuning():
    data_file = DATA_DIR / "ce_training_data.jsonl"
    if not os.path.exists(data_file):
        print(f"❌ Lỗi: Không tìm thấy {data_file}.")
        return

    print("📂 Nạp dữ liệu huấn luyện từ ổ cứng...")
    train_examples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            train_examples.append(InputExample(texts=[item["query"], item["passage"]], label=item["label"]))

    print(f"📦 Kіểm tra dữ liệu: Đã nạp thành công {len(train_examples)} cặp (Pos/Neg).")
    if len(train_examples) == 0:
        print("❌ LỖI: Dữ liệu bị rỗng! Bạn phải chạy lại file generate_ce_data.py")
        return

    # Hạ Batch Size xuống 8 để cực kỳ an toàn cho XLM-Roberta trên con A100 đang bị chiếm VRAM
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)

    print("🧹 Dọn dẹp VRAM trước khi nạp model...")
    torch.cuda.empty_cache()

    print("\n🔥 Nạp pre-trained Cross-Encoder...")
    model_path = str(MODELS_DIR / "reranker_model")
    
    target_device = "cuda:0" 
    model = CrossEncoder(model_path, num_labels=1, device=target_device)

    output_path = str(MODELS_DIR / "fine_tuned_ce_alqac")
    os.makedirs(output_path, exist_ok=True) # Tạo sẵn thư mục cho chắc ăn

    print(f"⚙️ Bắt đầu Fine-tuning (Vui lòng chờ khoảng 10-15 phút)...")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=2,
        warmup_steps=100,
        output_path=output_path, 
        show_progress_bar=True
    )
    
    # BƯỚC QUAN TRỌNG NHẤT: ÉP LƯU THỦ CÔNG
    print("\n💾 Đang ép mô hình ghi xuống ổ cứng...")
    model.save(output_path)
    
    print("\n" + "🌟"*25)
    print(f"✅ Đã Lưu Thành Công! Bạn có thể check bằng lệnh: ls -la {output_path}")
    print("🌟"*25)

if __name__ == "__main__":
    run_fine_tuning()