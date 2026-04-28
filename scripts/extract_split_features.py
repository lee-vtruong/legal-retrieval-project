import json
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import scipy.sparse
from tqdm import tqdm
from sentence_transformers import CrossEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval.hybrid import MultiViewHybridRetriever
from config import TRAIN_FILE, TEST_FILE, MODELS_DIR, DATA_DIR

def normalize_scores_array(raw_output):
    """Bộ giải mã bọc thép: Chuyển mọi định dạng về mảng Numpy 1 chiều"""
    # 1. Giải quyết thủ phạm gây lỗi: Scipy Sparse Matrix (BM25)
    if scipy.sparse.issparse(raw_output):
        return raw_output.toarray().flatten()
        
    # 2. Xử lý trường hợp bị gói trong Tuple
    if isinstance(raw_output, tuple):
        for item in raw_output:
            if hasattr(item, '__len__') and len(item) > 10:
                raw_output = item
                break
                
    # 3. Xử lý PyTorch Tensor (Dense/CE)
    if hasattr(raw_output, 'cpu'):
        return raw_output.cpu().detach().numpy().flatten()
        
    # 4. Fallback mặc định
    return np.array(raw_output).flatten()

def run_feature_extraction():
    clean_train_file = DATA_DIR / "train_clean_no_leak.json"
    active_train_file = clean_train_file if os.path.exists(clean_train_file) else TRAIN_FILE
    
    print(f"🚀 BƯỚC 1: Khởi động hệ thống trích xuất 6 Features...")
    print(f"📂 Đang sử dụng tập Train: {active_train_file}")
    
    retriever = MultiViewHybridRetriever(load_models=True)
    ce_model = CrossEncoder(str(MODELS_DIR / "fine_tuned_ce_alqac"), device="cuda:0")
    
    TOP_K = 30
    features_order = ["s_bm25", "s_dense", "s_ce", "p_ce", "u_uncertainty", "s_gat"]
    
    # ==========================================
    # PHẦN 1: TRÍCH XUẤT TẬP TRAIN
    # ==========================================
    print("\n🧠 BƯỚC 2: Trích xuất trên tập TRAIN...")
    with open(active_train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    X_train_list, y_train_list, qids = [], [], []

    for qid, item in enumerate(tqdm(train_data, desc="Train Extr")):
        query = item.get("text", "")
        gt_ids = [f"{rel['law_id']}_{rel['article_id']}" for rel in item.get("relevant_articles", [])]
        if not gt_ids: continue

        # --- ÉP MỌI ĐIỂM VỀ CHUẨN 1 CHIỀU ---
        bm25_scores = normalize_scores_array(retriever._get_sparse_scores(query))
        dense_scores = normalize_scores_array(retriever._get_dense_scores(query))
        combined_scores = normalize_scores_array(retriever._get_combined_scores(query)) 

        top_indices = np.argsort(combined_scores)[::-1][:TOP_K]
        pairs = [[query, retriever.chunks[idx]['text']] for idx in top_indices]
        
        ce_raw_scores = ce_model.predict(pairs, batch_size=32, convert_to_numpy=True)
        ce_probs = torch.nn.functional.softmax(torch.tensor(ce_raw_scores), dim=0).numpy()
        sorted_probs = np.sort(ce_probs)[::-1]
        u_val = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
        gat_norm = retriever._get_graph_scores(query)
        
        for i, idx in enumerate(top_indices):
            chunk = retriever.chunks[idx]
            art_key = f"{chunk['law_id']}_{chunk['article_id']}"
            gat_score = gat_norm[retriever.gat_mapping[art_key]] if art_key in retriever.gat_mapping else 0.0
            
            X_train_list.append([
                float(bm25_scores[idx]),   
                float(dense_scores[idx]),  
                float(ce_raw_scores[i]),   
                float(ce_probs[i]),        
                float(u_val),              
                float(gat_score)           
            ])
            y_train_list.append(1 if art_key in gt_ids else 0)
            qids.append(qid)

    X_train = pd.DataFrame(X_train_list, columns=features_order).fillna(0)
    y_train = np.array(y_train_list)

    # ==========================================
    # PHẦN 2: TRÍCH XUẤT TẬP TEST
    # ==========================================
    print("\n🧠 BƯỚC 3: Trích xuất trên tập TEST...")
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    test_processed = []
    
    for item in tqdm(test_data, desc="Test Extr"):
        query = item.get("text", "")
        gt_ids = [f"{rel['law_id']}_{rel['article_id']}" for rel in item.get("relevant_articles", [])]
        if not gt_ids: continue

        # --- ÉP MỌI ĐIỂM VỀ CHUẨN 1 CHIỀU ---
        bm25_scores = normalize_scores_array(retriever._get_sparse_scores(query))
        dense_scores = normalize_scores_array(retriever._get_dense_scores(query))
        combined_scores = normalize_scores_array(retriever._get_combined_scores(query)) 
        
        top_indices = np.argsort(combined_scores)[::-1][:TOP_K]
        pairs = [[query, retriever.chunks[idx]['text']] for idx in top_indices]
        
        ce_raw_scores = ce_model.predict(pairs, batch_size=32, convert_to_numpy=True)
        ce_probs = torch.nn.functional.softmax(torch.tensor(ce_raw_scores), dim=0).numpy()
        sorted_probs = np.sort(ce_probs)[::-1]
        u_val = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
        gat_norm = retriever._get_graph_scores(query)
        
        df_features = []
        mapping_idx = []
        
        for i, idx in enumerate(top_indices):
            chunk = retriever.chunks[idx]
            art_key = f"{chunk['law_id']}_{chunk['article_id']}"
            gat_score = gat_norm[retriever.gat_mapping[art_key]] if art_key in retriever.gat_mapping else 0.0
            
            df_features.append([
                float(bm25_scores[idx]), float(dense_scores[idx]), 
                float(ce_raw_scores[i]), float(ce_probs[i]), 
                float(u_val), float(gat_score)
            ])
            mapping_idx.append(art_key)

        test_processed.append({
            "gt": gt_ids, 
            "X": pd.DataFrame(df_features, columns=features_order).fillna(0), 
            "map": mapping_idx
        })

    # ==========================================
    # PHẦN 3: LƯU CACHE 6 FEATURES
    # ==========================================
    cache_file = DATA_DIR / "features_cache_6f.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump((X_train, y_train, qids, test_processed), f)
        
    print("\n" + "🌟"*25)
    print(f"✅ Đã trích xuất xong 6 Features SOTA!")
    print(f"📦 Cache được lưu an toàn tại: {cache_file}")
    print("🌟"*25)

if __name__ == "__main__":
    run_feature_extraction()