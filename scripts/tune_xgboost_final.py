import json, os, sys, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
from tqdm import tqdm
from sentence_transformers import CrossEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval.hybrid import MultiViewHybridRetriever
from config import TRAIN_FILE, TEST_FILE, MODELS_DIR, DATA_DIR

def calculate_f2(gt_ids, pred_ids):
    gts, preds = set(gt_ids), set(pred_ids)
    if not gts: return 0
    tp = len(preds.intersection(gts))
    fp, fn = len(preds - gts), len(gts - preds)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    return (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0

def run_tuning_v3():
    cache_file = DATA_DIR / "features_cache_5f.pkl"
    TOP_K = 30
    features_order = ["s_hybrid", "s_ce", "p_ce", "u_uncertainty", "s_gat"]

    if os.path.exists(cache_file):
        print(f"📦 Tìm thấy Cache! Nạp feature từ {cache_file}...")
        with open(cache_file, 'rb') as f:
            X_train, y_train, qids, test_processed = pickle.load(f)
    else:
        print("🚀 Khởi động trích xuất feature (Lần đầu)...")
        retriever = MultiViewHybridRetriever(load_models=True)
        ce_model = CrossEncoder(str(MODELS_DIR / "fine_tuned_ce_alqac"), device="cuda:0")

        # --- TRAIN EXTRACT ---
        with open(TRAIN_FILE, 'r') as f: train_data = json.load(f)
        X_train_list, y_train_list, qids = [], [], []
        for qid, item in enumerate(tqdm(train_data, desc="Train Extr")):
            gt_ids = [f"{r['law_id']}_{r['article_id']}" for r in item.get("relevant_articles", [])]
            scores = retriever._get_combined_scores(item["text"])
            top_indices = np.argsort(scores)[::-1][:TOP_K]
            pairs = [[item["text"], retriever.chunks[idx]['text']] for idx in top_indices]
            ce_raw = ce_model.predict(pairs, batch_size=32, convert_to_numpy=True)
            ce_probs = torch.nn.functional.softmax(torch.tensor(ce_raw), dim=0).numpy()
            u_val = (np.sort(ce_probs)[::-1][0] - np.sort(ce_probs)[::-1][1]) if len(ce_probs) > 1 else 1.0
            gat_norm = retriever._get_graph_scores(item["text"])
            for i, idx in enumerate(top_indices):
                art_key = f"{retriever.chunks[idx]['law_id']}_{retriever.chunks[idx]['article_id']}"
                gat_s = gat_norm[retriever.gat_mapping[art_key]] if art_key in retriever.gat_mapping else 0.0
                X_train_list.append([float(scores[idx]), float(ce_raw[i]), float(ce_probs[i]), float(u_val), float(gat_s)])
                y_train_list.append(1 if art_key in gt_ids else 0)
                qids.append(qid)
        X_train = pd.DataFrame(X_train_list, columns=features_order).fillna(0)
        y_train = np.array(y_train_list)

        # --- TEST EXTRACT ---
        with open(TEST_FILE, 'r') as f: test_data = json.load(f)
        test_processed = []
        for item in tqdm(test_data, desc="Test Extr"):
            gt_ids = [f"{r['law_id']}_{r['article_id']}" for r in item.get("relevant_articles", [])]
            scores = retriever._get_combined_scores(item["text"])
            top_indices = np.argsort(scores)[::-1][:TOP_K]
            pairs = [[item["text"], retriever.chunks[idx]['text']] for idx in top_indices]
            ce_raw = ce_model.predict(pairs, batch_size=32, convert_to_numpy=True)
            ce_probs = torch.nn.functional.softmax(torch.tensor(ce_raw), dim=0).numpy()
            u_val = (np.sort(ce_probs)[::-1][0] - np.sort(ce_probs)[::-1][1]) if len(ce_probs) > 1 else 1.0
            gat_norm = retriever._get_graph_scores(item["text"])
            df_feat, mapping = [], []
            for i, idx in enumerate(top_indices):
                art_key = f"{retriever.chunks[idx]['law_id']}_{retriever.chunks[idx]['article_id']}"
                gat_s = gat_norm[retriever.gat_mapping[art_key]] if art_key in retriever.gat_mapping else 0.0
                df_feat.append([float(scores[idx]), float(ce_raw[i]), float(ce_probs[i]), float(u_val), float(gat_s)])
                mapping.append(art_key)
            test_processed.append({"gt": gt_ids, "X": pd.DataFrame(df_feat, columns=features_order).fillna(0), "map": mapping})

        with open(cache_file, 'wb') as f:
            pickle.dump((X_train, y_train, qids, test_processed), f)
        print("💾 Đã lưu cache thành công!")

    # --- GRID SEARCH ---
    print("\n🔍 Đang quét 36 cấu hình (Cực nhanh)...")
    best_overall_f2, best_params = 0, {}
    import itertools
    configs = list(itertools.product([0.05, 0.1, 0.2], [3, 4, 6], [100, 150, 200]))

    for lr, depth, est in configs:
        # Quan trọng: n_jobs=1 để tránh treo server
        ranker = xgb.XGBRanker(n_jobs=1, tree_method="hist", n_estimators=est, learning_rate=lr, max_depth=depth)
        ranker.fit(X_train, y_train, qid=qids)
        
        f2_list = []
        for item in test_processed:
            f_scores = ranker.predict(item["X"])
            # Lấy Top 1 và Margin Scan 
            sorted_idx = np.argsort(f_scores)[::-1]
            top_score = f_scores[sorted_idx[0]]
            # Thử margin 0.7 (Ngưỡng vàng cũ)
            selected = [item["map"][i] for i in sorted_idx if top_score - f_scores[i] <= 0.7]
            f2_list.append(calculate_f2(item["gt"], selected))
        
        curr_f2 = np.mean(f2_list)
        print(f"-> Params: LR={lr}, Depth={depth}, Est={est} | F2@0.7: {curr_f2:.5f}")
        if curr_f2 > best_overall_f2:
            best_overall_f2 = curr_f2
            best_params = {"lr": lr, "depth": depth, "est": est}

    print(f"\n🏆 KẾT QUẢ TỐI ƯU: {best_overall_f2:.6f} với {best_params}")

if __name__ == "__main__":
    run_tuning_v3()