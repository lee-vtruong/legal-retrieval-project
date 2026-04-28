import json
import os
import sys
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_FILE, MODELS_DIR

class LawDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def train_law_classifier():
    print("📂 Đọc dữ liệu tập Train...")
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    texts = []
    labels = []
    law2id = {}

    print("🧠 Đang trích xuất nhãn Law ID...")
    for item in train_data:
        gt_articles = item.get("relevant_articles", [])
        if not gt_articles: continue
        
        # Lấy law_id của đáp án đầu tiên làm nhãn (Đa số các câu ALQAC gom chung 1 luật)
        law_id = gt_articles[0]["law_id"]
        
        if law_id not in law2id:
            law2id[law_id] = len(law2id)
            
        texts.append(item["text"])
        labels.append(law2id[law_id])

    num_labels = len(law2id)
    print(f"📚 Phát hiện {num_labels} bộ luật khác nhau trong tập Train.")

    # Lưu lại mapping để lúc inference còn biết index nào là luật nào
    classifier_dir = MODELS_DIR / "law_classifier_phobert"
    os.makedirs(classifier_dir, exist_ok=True)
    with open(classifier_dir / "law2id.json", "w", encoding="utf-8") as f:
        json.dump(law2id, f, ensure_ascii=False, indent=4)

    print("⚙️ Khởi tạo Tokenizer (vinai/phobert-base-v2)...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)
    
    dataset = LawDataset(encodings, labels)

    print("🔥 Nạp mô hình PhoBERT-base...")
    # Tắt cảnh báo weight initialization vì chúng ta đang train task mới
    model = AutoModelForSequenceClassification.from_pretrained(
        "vinai/phobert-base-v2", 
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir=str(classifier_dir / "checkpoints"),
        num_train_epochs=5,              # 5 epochs là đủ để hội tụ cho text classification
        per_device_train_batch_size=4,  # Chống OOM
        learning_rate=2e-5,
        logging_steps=50,
        save_strategy="no",              # Lưu thủ công ở cuối cho gọn
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("🚀 Bắt đầu Fine-tuning PhoBERT Law Classifier...")
    trainer.train()

    print("\n💾 Đang lưu mô hình hoàn chỉnh...")
    model.save_pretrained(str(classifier_dir))
    tokenizer.save_pretrained(str(classifier_dir))
    
    print("\n" + "🌟"*25)
    print(f"✅ Đã huấn luyện xong Law Classifier! Lưu tại: {classifier_dir}")
    print("🌟"*25)

if __name__ == "__main__":
    train_law_classifier()