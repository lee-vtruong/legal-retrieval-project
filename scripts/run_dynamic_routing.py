#!/usr/bin/env python3
"""
Qwen3-14B Scorer + XGBoost Meta-Ranker.
LLM chấm điểm 0‑10 từng bài luật, bổ sung làm feature thứ 6.
"""

import os, sys, json, pickle, re, time, argparse
import numpy as np, pandas as pd, xgboost as xgb
from tqdm import tqdm

# Fix vLLM
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['NCCL_SHM_DISABLE'] = '1'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TRAIN_FILE, TEST_FILE, MODELS_DIR
from src.retrieval.hybrid import MultiViewHybridRetriever

# =================== Tiện ích ===================
def calculate_f2(gt_ids, pred_ids):
    gts, preds = set(gt_ids), set(pred_ids)
    if not gts: return 0.0
    tp = len(preds & gts)
    fp = len(preds - gts)
    fn = len(gts - preds)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    return (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0

def aggregate_articles(scores, ids):
    d = {}
    for art, s in zip(ids, scores):
        if art not in d or s > d[art]:
            d[art] = s
    return sorted(d.items(), key=lambda x: x[1], reverse=True)

def dynamic_margin(art_list, margin=0.7):
    if not art_list:
        return []
    top_score = art_list[0][1]
    return [a for a, s in art_list if top_score - s <= margin]

# =================== LLM Scorer ===================
def build_scorer_prompt(query, candidates):
    """
    candidates: list of (art_id, text) up to 15.
    Yêu cầu LLM trả JSON: {"scores": {"id1": score1, ...}}
    """
    cand_str = "\n".join([f"{i+1}. {art_id}: {text[:250]}" for i, (art_id, text) in enumerate(candidates)])
    prompt = f"""<|im_start|>system
Bạn là trợ lý pháp lý. Cho một câu hỏi và danh sách điều luật ứng viên, hãy đánh giá mức độ liên quan của mỗi điều luật với câu hỏi trên thang điểm 0 (không liên quan) đến 10 (rất liên quan). Trả về JSON có key "scores" ánh xạ từ ID điều luật sang điểm số.
Ví dụ:
{{"scores": {{"Bộ luật Dân sự_123": 8, "Luật Hình sự_45": 2}}}}
<|im_end|>
<|im_start|>user
Câu hỏi: {query}
Danh sách ứng viên:
{cand_str}
<|im_end|>
<|im_start|>assistant
{{"scores": {{"""  # prompt dừng để model tự viết tiếp
    return prompt

def extract_scores(text, id_list):
    """Tìm map scores từ output LLM."""
    # Tìm khối JSON chứa "scores"
    match = re.search(r'\{[^{}]*"scores"\s*:\s*\{[^{}]*\}\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            score_map = data["scores"]
            # Chuyển key về đúng định dạng, có thể LLM thêm dấu cách
            clean = {}
            for k, v in score_map.items():
                # Bỏ dấu ngoặc kép thừa và chuẩn hóa
                clean_id = k.strip().replace('"', '')
                if clean_id in id_list:
                    clean[clean_id] = float(v)/10.0  # normalize về 0-1
            return clean
        except:
            pass
    return {}

# =================== Main ===================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=5)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--no_cache', action='store_true', help='Không dùng cache LLM cũ')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print(f"🎯 GPU {args.gpu} – LLM Scorer (Qwen3-14B)")

    # 1. Load retriever
    print("📚 Nạp Hybrid Retriever...")
    retriever = MultiViewHybridRetriever(load_models=True)

    # 2. Load dữ liệu train/test
    with open(TRAIN_FILE, 'r') as f: train_data = json.load(f)
    with open(TEST_FILE, 'r') as f: test_data = json.load(f)

    if not args.full:
        train_data = train_data[:100]
        test_data = test_data[:20]
        print("⚠️  Chế độ test nhanh với tập con.")

    # 3. Khởi tạo LLM (14B)
    print("🤖 Khởi tạo Qwen3-14B...")
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen3-14B", trust_remote_code=True,
              gpu_memory_utilization=0.6, disable_log_stats=True)
    samp = SamplingParams(temperature=0.0, max_tokens=512)

    # 4. Tính s_llm cho train & test (có cache nếu muốn)
    cache_path = DATA_DIR / "llm_scores_cache.pkl"
    if os.path.exists(cache_path) and not args.no_cache:
        with open(cache_path, 'rb') as f:
            train_sllm, test_sllm = pickle.load(f)
        print("📦 Đã nạp cache điểm LLM.")
    else:
        train_sllm, test_sllm = [], []
        for split_name, data_list in [("Train", train_data), ("Test", test_data)]:
            print(f"📊 Đang chấm điểm cho tập {split_name}...")
            for item in tqdm(data_list, desc=split_name):
                query = item["text"]
                # Lấy top-15 candidates (đảm bảo recall cao)
                scores = retriever._get_combined_scores(query)
                top_indices = np.argsort(scores)[::-1][:15]
                candidates = []
                seen_ids = set()
                for idx in top_indices:
                    chunk = retriever.chunks[idx]
                    art_id = f"{chunk['law_id']}_{chunk['article_id']}"
                    if art_id not in seen_ids:
                        seen_ids.add(art_id)
                        candidates.append((art_id, chunk['text']))
                    if len(candidates) == 15:
                        break
                prompt = build_scorer_prompt(query, candidates)
                try:
                    outputs = llm.generate([prompt], samp, use_tqdm=False)
                    raw = outputs[0].outputs[0].text.strip()
                except Exception as e:
                    raw = f"ERROR: {e}"
                # Parse scores
                score_map = extract_scores(raw, [c[0] for c in candidates])
                # Nếu parse lỗi, gán 0.5 cho tất cả
                if not score_map:
                    score_map = {art_id: 0.5 for art_id, _ in candidates}
                # Lưu cho query này
                if split_name == "Train":
                    train_sllm.append(score_map)
                else:
                    test_sllm.append(score_map)
        # Lưu cache
        with open(cache_path, 'wb') as f:
            pickle.dump((train_sllm, test_sllm), f)
        print("💾 Đã lưu cache điểm LLM.")

    # 5. Trích xuất feature 6 cột cho XGBoost
    print("🧠 Trích xuất feature XGBoost...")
    from sentence_transformers import CrossEncoder
    ce_model = CrossEncoder(str(MODELS_DIR / "fine_tuned_ce_alqac"), device="cuda:0")
    TOP_K = 30
    features_order = ["s_hybrid", "s_ce", "p_ce", "u_uncertainty", "s_gat", "s_llm"]

    def extract_features(data_list, llm_scores_list):
        X_list, y_list, qids, test_processed = [], [], [], []
        for qid, item in enumerate(tqdm(data_list, desc="Extract")):
            query = item["text"]
            gt_ids = [f"{r['law_id']}_{r['article_id']}" for r in item.get("relevant_articles", [])]
            scores = retriever._get_combined_scores(query)
            top_indices = np.argsort(scores)[::-1][:TOP_K]
            pairs = [[query, retriever.chunks[idx]['text']] for idx in top_indices]
            ce_raw = ce_model.predict(pairs, batch_size=4, convert_to_numpy=True)
            ce_probs = torch.softmax(torch.tensor(ce_raw), dim=0).numpy()
            u_val = (np.sort(ce_probs)[::-1][0] - np.sort(ce_probs)[::-1][1]) if len(ce_probs) > 1 else 1.0
            gat_norm = retriever._get_graph_scores(query)
            # Lấy điểm LLM (có thể trả về 0.5 nếu không có)
            llm_map = llm_scores_list[qid]
            df_feat, mapping = [], []
            for i, idx in enumerate(top_indices):
                art_key = f"{retriever.chunks[idx]['law_id']}_{retriever.chunks[idx]['article_id']}"
                gat_s = gat_norm[retriever.gat_mapping[art_key]] if art_key in retriever.gat_mapping else 0.0
                s_llm = llm_map.get(art_key, 0.5)  # mặc định 0.5 nếu thiếu
                df_feat.append([float(scores[idx]), float(ce_raw[i]), float(ce_probs[i]),
                                float(u_val), float(gat_s), float(s_llm)])
                mapping.append(art_key)
            # Nếu là tập test thì lưu cấu trúc để đánh giá
            if qid < len(test_processed):
                test_processed.append({"gt": gt_ids, "X": pd.DataFrame(df_feat, columns=features_order).fillna(0), "map": mapping})
            # Nếu tập train: thu thập dữ liệu huấn luyện
            X_list.extend(df_feat)
            y_list.extend([1 if art_key in gt_ids else 0 for art_key in mapping])
            qids.extend([qid]*len(mapping))
        return np.array(X_list), np.array(y_list), qids, test_processed

    # Train
    X_train, y_train, qids_train, _ = extract_features(train_data, train_sllm)
    # Test
    X_test_features, y_test_features, qids_test, test_processed = extract_features(test_data, test_sllm)

    # 6. Huấn luyện XGBoost với 6 features
    print("🚀 Huấn luyện XGBRanker (6 features)...")
    ranker = xgb.XGBRanker(n_jobs=1, tree_method="hist", objective="rank:ndcg",
                           n_estimators=150, learning_rate=0.05, max_depth=6)
    ranker.fit(X_train, y_train, qid=qids_train)

    # Đánh giá Dynamic Margin 0.7
    MARGIN = 0.7
    f2_scores = []
    for item in test_processed:
        f_scores = ranker.predict(item["X"])
        art_list = aggregate_articles(f_scores, item["map"])
        preds = dynamic_margin(art_list, MARGIN)
        f2_scores.append(calculate_f2(item["gt"], preds))

    final_f2 = np.mean(f2_scores)
    print("\n" + "=" * 60)
    print(f"🌟 F2 với LLM Scorer (Qwen3-14B + 6 features): {final_f2:.6f}")
    print("=" * 60)