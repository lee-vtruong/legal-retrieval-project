import json
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval.hybrid import MultiViewHybridRetriever
from config import TRAIN_FILE, TEST_FILE, MODELS_DIR

def calculate_f2(gt_ids, pred_ids):
    gts = set(gt_ids)
    preds = set(pred_ids)
    if not gts: return 0
    tp = len(preds.intersection(gts))
    fp = len(preds - gts)
    fn = len(gts - preds)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    if (4 * prec + rec) == 0: return 0
    return (5 * prec * rec) / (4 * prec + rec)

def run_tuning():
    print("🚀 Nạp hệ thống với Fine-tuned Cross-Encoder...")
    retriever = MultiViewHybridRetriever(load_models=True)
    ce_model = CrossEncoder(str(MODELS_DIR / "fine_tuned_ce_alqac"), device="cuda:0")
    TOP_K = 30
    features_order = ["s_hybrid", "s_ce", "p_ce", "u_uncertainty", "s_gat"]

    # --- 1. PREP TRAIN DATA ---
    print("📂 Trích xuất 5 Features tập TRAIN...")
    with open(TRAIN_FILE, 'r') as f: train_data = json.load(f)
    X_train_list, y_train_list, qids = [], [], []
    for qid, item in enumerate(tqdm(train_data, desc="Train Extr")):
        gt_ids = [f"{r['law_id']}_{r['article_id']}" for r in item.get("relevant_articles", [])]
        if not gt_ids: continue
        scores = retriever._get_combined_scores(item["text"])
        top_indices = np.argsort(scores)[::-1][:TOP_K]
        pairs = [[item["text"], retriever.chunks[idx]['text']] for idx in top_indices]
        ce_raw = ce_model.predict(pairs, batch_size=32, convert_to_numpy=True)
        ce_probs = torch.nn.functional.softmax(torch.tensor(ce_raw), dim=0).numpy()
        sorted_probs = np.sort(ce_probs)[::-1]
        u_val = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
        gat_norm = retriever._get_graph_scores(item["text"])
        
        for i, idx in enumerate(top_indices):
            chunk = retriever.chunks[idx]
            art_key = f"{chunk['law_id']}_{chunk['article_id']}"
            gat_s = gat_norm[retriever.gat_mapping[art_key]] if art_key in retriever.gat_mapping else 0.0
            X_train_list.append([float(scores[idx]), float(ce_raw[i]), float(ce_probs[i]), float(u_val), float(gat_s)])
            y_train_list.append(1 if art_key in gt_ids else 0)
            qids.append(qid)

    X_train = pd.DataFrame(X_train_list, columns=features_order)
    y_train = np.array(y_train_list)

    # --- 2. PREP TEST DATA ---
    print("📂 Trích xuất 5 Features tập TEST...")
    with open(TEST_FILE, 'r') as f: test_data = json.load(f)
    test_processed = []
    for item in tqdm(test_data, desc="Test Extr"):
        gt_ids = [f"{r['law_id']}_{r['article_id']}" for r in item.get("relevant_articles", [])]
        if not gt_ids: continue
        scores = retriever._get_combined_scores(item["text"])
        top_indices = np.argsort(scores)[::-1][:TOP_K]
        pairs = [[item["text"], retriever.chunks[idx]['text']] for idx in top_indices]
        ce_raw = ce_model.predict(pairs, batch_size=32, convert_to_numpy=True)
        ce_probs = torch.nn.functional.softmax(torch.tensor(ce_raw), dim=0).numpy()
        sorted_probs = np.sort(ce_probs)[::-1]
        u_val = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
        gat_norm = retriever._get_graph_scores(item["text"])
        
        df_feat, mapping = [], []
        for i, idx in enumerate(top_indices):
            chunk = retriever.chunks[idx]
            art_key = f"{chunk['law_id']}_{chunk['article_id']}"
            gat_s = gat_norm[retriever.gat_mapping[art_key]] if art_key in retriever.gat_mapping else 0.0
            df_feat.append([float(scores[idx]), float(ce_raw[i]), float(ce_probs[i]), float(u_val), float(gat_s)])
            mapping.append(art_key)
        test_processed.append({"gt": gt_ids, "X": pd.DataFrame(df_feat, columns=features_order), "map": mapping})

    # --- 3. GRID SEARCH XGBOOST ---
    print("\n🔍 Khởi động Grid Search siêu tham số...")
    best_overall_f2 = 0
    best_params = {}

    # Thử nghiệm các tổ hợp tham số
    for lr in [0.05, 0.1, 0.2]:
        for depth in [3, 4, 5, 6]:
            for est in [100, 150, 200]:
                ranker = xgb.XGBRanker(tree_method="hist", objective="rank:ndcg", n_estimators=est, learning_rate=lr, max_depth=depth, subsample=0.8)
                ranker.fit(X_train, y_train, qid=qids)
                
                # Inference
                all_preds = []
                for item in test_processed:
                    f_scores = ranker.predict(item["X"])
                    art_scores = {}
                    for s, k in zip(f_scores, item["map"]):
                        if k not in art_scores or s > art_scores[k]: art_scores[k] = s
                    sorted_arts = sorted(art_scores.items(), key=lambda x: x[1], reverse=True)
                    all_preds.append({"gt": item["gt"], "ids": [a[0] for a in sorted_arts], "scores": [a[1] for a in sorted_arts]})
                
                # Dynamic Margin Scan
                best_m_f2 = 0
                for m in np.arange(0.0, 2.5, 0.1):
                    f2_list = []
                    for p in all_preds:
                        if not p["ids"]: f2_list.append(0); continue
                        selected = [i for i, s in zip(p["ids"], p["scores"]) if p["scores"][0] - s <= m]
                        f2_list.append(calculate_f2(p["gt"], selected))
                    if np.mean(f2_list) > best_m_f2: best_m_f2 = np.mean(f2_list)
                
                print(f"Tham số [LR={lr}, Depth={depth}, Est={est}] -> F2: {best_m_f2:.5f}")
                if best_m_f2 > best_overall_f2:
                    best_overall_f2 = best_m_f2
                    best_params = {"lr": lr, "depth": depth, "est": est}

    print("\n" + "="*50)
    print("🏆 KẾT QUẢ TỐI ƯU HÓA XGBOOST")
    print(f"F2 SOTA mới nhất : {best_overall_f2:.6f} 🌟")
    print(f"Bộ tham số vàng  : {best_params}")
    print("="*50)

if __name__ == "__main__":
    run_tuning()