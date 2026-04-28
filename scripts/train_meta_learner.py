import pandas as pd
import xgboost as xgb
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, MODELS_DIR

def train_meta():
    feature_path = DATA_DIR / "processed" / "meta_train_features.csv"
    group_path = DATA_DIR / "processed" / "meta_train_groups.txt"
    
    if not os.path.exists(feature_path):
        print("❌ Lỗi: Không tìm thấy meta_train_features.csv. Hãy chạy extract_meta_features.py trước!")
        return

    print(f"🚀 Nạp dữ liệu từ {feature_path}...")
    df = pd.read_csv(feature_path)
    
    # 1. Chọn đặc trưng (Features)
    features = ["s_hybrid", "s_ce", "p_ce", "u_uncertainty", "s_gat"]
    X = df[features]
    y = df["label"]
    
    groups = []
    with open(group_path, "r") as f:
        for line in f:
            groups.append(int(line.strip()))

    # 2. Khởi tạo XGBRanker
    print("🧠 Bắt đầu huấn luyện Meta-Learner (XGBRanker)...")
    ranker = xgb.XGBRanker(
        tree_method="hist", 
        device="cuda:5", # Khai thác GPU 5
        objective="rank:pairwise",
        lambdarank_pair_method="topk",
        eta=0.1,
        max_depth=6,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    # 3. Huấn luyện
    ranker.fit(X, y, group=groups, verbose=True)

    # 4. Lưu Model
    model_save_path = MODELS_DIR / "xgboost_ranker.json"
    os.makedirs(MODELS_DIR, exist_ok=True)
    ranker.save_model(str(model_save_path))
    
    print(f"✅ Đã luyện xong Stage 5! Model lưu tại: {model_save_path}")

    # 5. Phân tích độ quan trọng của đặc trưng (Feature Importance)
    importance = ranker.feature_importances_
    print("\n📊 MỨC ĐỘ QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG (Dùng cho Báo Cáo):")
    for feat, imp in zip(features, importance):
        print(f"   - {feat:15s}: {imp:.4f}")

if __name__ == "__main__":
    train_meta()