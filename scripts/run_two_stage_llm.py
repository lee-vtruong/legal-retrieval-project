import json
import os
import sys
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from vllm import LLM, SamplingParams

# Import từ hệ thống Clean Architecture
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR
from src.retrieval.hybrid import MultiViewHybridRetriever

def extract_json_array(text):
    try:
        matches = re.findall(r'\[.*?\]', text, re.DOTALL)
        if matches:
            return json.loads(matches[-1])
        return []
    except:
        return []

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

def run_two_stage_pipeline():
    print("🚀 Đang nạp Dữ liệu văn bản luật...")
    retriever = MultiViewHybridRetriever(load_models=False) 
    article_texts = {}
    for chunk in retriever.chunks:
        key = f"{chunk['law_id']}_{chunk['article_id']}"
        if key not in article_texts:
            article_texts[key] = ""
        article_texts[key] += chunk['text'] + "\n"

    print("📂 Đọc kết quả Top 10 từ hệ thống...")
    submission_file = DATA_DIR / "results" / "final_submission.json"
    with open(submission_file, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)

    print("🧠 Khởi tạo vLLM Engine (Qwen2.5-7B-Instruct)...")
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True, gpu_memory_utilization=0.7)
    
    sampling_params_stage1 = SamplingParams(temperature=0.0, max_tokens=10)
    sampling_params_stage2 = SamplingParams(temperature=0.0, max_tokens=400)

    TOP_K = 10
    
    # --- STAGE 1: POINTWISE FILTER ---
    print("\n" + "="*40 + "\n🚀 STAGE 1: CHẠY POINTWISE FILTER\n" + "="*40)
    stage1_prompts = []
    stage1_mapping = [] 
    
    for q_idx, item in enumerate(predictions_data):
        query = item["text"]
        preds = item.get("predictions", [])[:TOP_K]
        for p in preds:
            art_id = f"{p['law_id']}_{p['article_id']}"
            text = " ".join(article_texts.get(art_id, "").split()[:250])
            prompt = f"""<|im_start|>system
Bạn là AI phân tích pháp lý. Đọc NHẬN ĐỊNH/CÂU HỎI và VĂN BẢN.
Hỏi: VĂN BẢN có chứa quy định pháp luật để kiểm chứng tính Đúng/Sai của NHẬN ĐỊNH/CÂU HỎI này không?
CHỈ đáp duy nhất: "Có" hoặc "Không".<|im_end|>
<|im_start|>user
NHẬN ĐỊNH/CÂU HỎI: {query}
VĂN BẢN: {text}<|im_end|>
<|im_start|>assistant
"""
            stage1_prompts.append(prompt)
            stage1_mapping.append((q_idx, art_id))

    stage1_outputs = llm.generate(stage1_prompts, sampling_params_stage1)
    filtered_candidates = {i: [] for i in range(len(predictions_data))}
    for out, (q_idx, art_id) in zip(stage1_outputs, stage1_mapping):
        if "có" in out.outputs[0].text.strip().lower():
            filtered_candidates[q_idx].append(art_id)

    # --- STAGE 2: LISTWISE RERANKING ---
    print("\n" + "="*40 + "\n🚀 STAGE 2: CHẠY LISTWISE RERANKING\n" + "="*40)
    stage2_prompts, stage2_indices, final_predictions = [], [], {}
    
    for q_idx, item in enumerate(predictions_data):
        candidates = filtered_candidates[q_idx]
        if not candidates:
            final_predictions[q_idx] = [f"{item['predictions'][0]['law_id']}_{item['predictions'][0]['article_id']}"]
        elif len(candidates) == 1:
            final_predictions[q_idx] = candidates
        else:
            context_str = "".join([f"\n[ID: {cid}]\n{ ' '.join(article_texts.get(cid, '').split()[:300]) }\n" for cid in candidates])
            prompt = f"<|im_start|>system\nBạn là Thẩm phán. Phân tích rồi chốt JSON ID đúng.\n<|im_end|>\n<|im_start|>user\nCÂU HỎI: {item['text']}\nDANH SÁCH:{context_str}\n<|im_end|>\n<|im_start|>assistant\nPhân tích:"
            stage2_prompts.append(prompt)
            stage2_indices.append(q_idx)

    if stage2_prompts:
        stage2_outputs = llm.generate(stage2_prompts, sampling_params_stage2)
        for out, q_idx in zip(stage2_outputs, stage2_indices):
            final_predictions[q_idx] = [str(pid).strip() for pid in extract_json_array(out.outputs[0].text)]

    # --- LƯU KẾT QUẢ VÀ TÍNH METRICS ---
    results_log, f2_scores = [], []
    for q_idx, item in enumerate(predictions_data):
        gt_ids = item.get("relevant_articles", [])
        pred_ids = final_predictions.get(q_idx, [])
        f2 = calculate_f2(gt_ids, pred_ids)
        f2_scores.append(f2)
        results_log.append({
            "question_id": item.get("question_id"),
            "query": item["text"],
            "ground_truth": gt_ids,
            "stage1_filtered": filtered_candidates[q_idx],
            "final_predictions": pred_ids,
            "f2_score": f2
        })

    # Lưu file
    os.makedirs(DATA_DIR / "results", exist_ok=True)
    timestamp = datetime.now().strftime('%m%d_%H%M')
    
    with open(DATA_DIR / "results" / f"two_stage_details_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=4)
        
    summary_df = pd.DataFrame([{"Configuration": "GAM-LLM Two-Stage", "F2-macro": np.mean(f2_scores)}])
    summary_df.to_csv(DATA_DIR / "results" / f"two_stage_metrics_{timestamp}.csv", index=False)

    print("\n" + "🌟"*25 + f"\n🏆 KẾT QUẢ: {np.mean(f2_scores):.6f}\n" + "🌟"*25)
    print(f"✅ Đã lưu chi tiết tại: data/results/two_stage_details_{timestamp}.json")

if __name__ == "__main__":
    run_two_stage_pipeline()