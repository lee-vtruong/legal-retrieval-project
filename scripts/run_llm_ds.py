#!/usr/bin/env python3
"""
LLM Reason-and-Extract với vLLM engine V0 – Đã Fix lỗi khởi tạo.
"""

import os

# 1. Tắt V1 bằng đúng cờ chuẩn của vLLM hiện tại
os.environ['VLLM_USE_V1'] = '0'

# 2. Ép phương thức đa tiến trình thành 'spawn' để không đụng chạm với XGBoost/CUDA
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# 3. Ép giới hạn chia sẻ bộ nhớ (chống sập Core trên các container/server)
os.environ['NCCL_SHM_DISABLE'] = '1'

import sys, json, pickle, re, gc, argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

# ====================== Tiện ích ======================
def calculate_f2(gt_ids, pred_ids):
    gts, preds = set(gt_ids), set(pred_ids)
    if not gts: return 0.0
    tp = len(preds.intersection(gts))
    fp = len(preds - gts)
    fn = len(gts - preds)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    return (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0

def build_prompt(query, cand_texts, top_art_ids):
    cand_list = []
    for i, art_id in enumerate(top_art_ids):
        text = cand_texts.get(art_id, "")[:300]
        cand_list.append(f"[{i+1}] {art_id}: {text}")
    cand_str = "\n".join(cand_list)
    return f"""<|im_start|>system
Bạn là thẩm phán AI. Cho một nhận định pháp lý và danh sách các điều luật ứng viên. Hãy suy luận từng bước để xác định điều luật nào là căn cứ pháp lý trực tiếp cho nhận định đó. Sau đó xuất kết quả dưới dạng JSON: {{"relevant_articles": ["<law_id>_<article_id>"]}}
<|im_end|>
<|im_start|>user
Nhận định: {query}
Các điều luật ứng viên:
{cand_str}
Hãy suy luận cẩn thận và chỉ chọn những điều luật thực sự đúng.
<|im_end|>
<|im_start|>assistant
Suy luận:"""

def extract_json(text):
    pattern = r'\{[^{}]*"relevant_articles"[^{}]*\}'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        try:
            return json.loads(matches[-1])
        except json.JSONDecodeError:
            pass
    return None

# ====================== Main ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=5)
    parser.add_argument('--full', action='store_true', help='Chạy toàn bộ test set (mặc định 10 câu)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print(f"🎯 GPU {args.gpu}, vLLM engine V0")

    # 1. Load cache
    cache_path = DATA_DIR / "features_cache_5f_with_text.pkl"
    with open(cache_path, 'rb') as f:
        X_train, y_train, qids, test_processed = pickle.load(f)

    if not args.full:
        test_processed = test_processed[:10]
    print(f"📊 {len(test_processed)} questions")

    # 2. Train XGBoost (CPU)
    print("🚀 Training XGBoost...")
    ranker = xgb.XGBRanker(n_jobs=1, tree_method="hist", n_estimators=150,
                           learning_rate=0.05, max_depth=6)
    ranker.fit(X_train, y_train, qid=qids)

    # 3. Baseline top-1
    f2_top1 = []
    for item in test_processed:
        f_scores = ranker.predict(item["X"])
        top1 = [item["map"][np.argmax(f_scores)]]
        f2_top1.append(calculate_f2(item["gt"], top1))
    baseline = np.mean(f2_top1)
    print(f"📊 XGBoost Top-1 F2: {baseline:.6f}")

    # 4. Giải phóng CPU (không import torch)
    gc.collect()

    # 5. Khởi tạo vLLM V0
    print("🤖 Khởi tạo Qwen2.5-7B-Instruct (V0 engine)...")
    from vllm import LLM, SamplingParams

    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct",
              trust_remote_code=True,
              gpu_memory_utilization=0.5,
              disable_log_stats=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    # 6. Chạy LLM
    f2_llm = []
    for item in tqdm(test_processed, desc="LLM"):
        f_scores = ranker.predict(item["X"])
        sorted_idx = np.argsort(f_scores)[::-1]
        top_art_ids = [item["map"][i] for i in sorted_idx[:10]]

        prompt = build_prompt(item["query"], item["cand_texts"], top_art_ids)

        try:
            outputs = llm.generate([prompt], sampling_params)
            raw_output = outputs[0].outputs[0].text
        except Exception as e:
            raw_output = f"ERROR: {e}"

        parsed = extract_json(raw_output) if raw_output else None
        if parsed and "relevant_articles" in parsed:
            pred_ids = parsed["relevant_articles"]
        else:
            pred_ids = [item["map"][sorted_idx[0]]]

        f2_llm.append(calculate_f2(item["gt"], pred_ids[:1]))

    final_f2 = np.mean(f2_llm)
    print("\n" + "=" * 60)
    print(f"🌟 LLM F2-macro: {final_f2:.6f}")
    print(f"📈 So với XGBoost Top-1: {final_f2 - baseline:+.6f}")
    print("=" * 60)