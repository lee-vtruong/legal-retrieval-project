import json
import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval.hybrid import MultiViewHybridRetriever
from config import DATA_DIR, TEST_FILE, MODELS_DIR

def calculate_f2(gt_ids, pred_ids):
    gts, preds = set(gt_ids), set(pred_ids)
    if not gts: return 0
    tp = len(preds.intersection(gts))
    fp, fn = len(preds - gts), len(gts - preds)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    return (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0

def build_fewshot_cot_prompt(query, document):
    """
    Kỹ thuật Few-shot CoT: Ép Qwen học theo 2 mẫu ví dụ trước khi trả lời câu thật.
    """
    system_msg = "Bạn là Thẩm phán Tòa án Tối cao Việt Nam. Nhiệm vụ của bạn là đánh giá tính liên quan của văn bản pháp luật đối với câu hỏi. Bạn phải lập luận từng bước và chốt lại bằng định dạng 'RELEVANCE_SCORE: 1' hoặc 'RELEVANCE_SCORE: 0'."
    
    # Ví dụ 1: True Positive (Đúng hoàn cảnh, đúng đối tượng)
    user_ex1 = "Câu hỏi: 'Mức phạt lỗi chạy quá tốc độ 5 đến 10 km/h đối với ô tô?'\nVăn bản: 'Điều 5. Xử phạt người điều khiển xe ô tô... Phạt tiền từ 800.000 đến 1.000.000 đồng đối với người điều khiển xe chạy quá tốc độ quy định từ 05 km/h đến dưới 10 km/h.'"
    asst_ex1 = "Lập luận: Văn bản quy định rõ mức phạt tiền từ 800.000 đến 1.000.000 đồng cho hành vi chạy quá tốc độ từ 05 km/h đến dưới 10 km/h. Đối tượng áp dụng là 'xe ô tô', hoàn toàn khớp với câu hỏi. Văn bản này trực tiếp giải quyết vấn đề.\nRELEVANCE_SCORE: 1"

    # Ví dụ 2: Hard Negative (Trùng từ khóa nhưng sai đối tượng)
    user_ex2 = "Câu hỏi: 'Mức phạt lỗi chạy quá tốc độ 5 đến 10 km/h đối với ô tô?'\nVăn bản: 'Điều 6. Xử phạt người điều khiển xe mô tô, xe gắn máy... Phạt tiền từ 300.000 đến 400.000 đồng đối với hành vi điều khiển xe chạy quá tốc độ quy định từ 05 km/h đến dưới 10 km/h.'"
    asst_ex2 = "Lập luận: Văn bản có đề cập đến lỗi chạy quá tốc độ từ 05 đến 10 km/h. Tuy nhiên, đối tượng áp dụng ở đây là 'xe mô tô, xe gắn máy', trong khi câu hỏi cụ thể hỏi về 'xe ô tô'. Do sai lệch đối tượng, văn bản này không áp dụng được.\nRELEVANCE_SCORE: 0"

    # Câu hỏi thực tế
    user_real = f"Câu hỏi: '{query}'\nVăn bản: '{document}'\nHãy lập luận và chốt điểm theo đúng định dạng."

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_ex1},
        {"role": "assistant", "content": asst_ex1},
        {"role": "user", "content": user_ex2},
        {"role": "assistant", "content": asst_ex2},
        {"role": "user", "content": user_real}
    ]

def run_qwen_fewshot():
    print("🚀 BƯỚC 1: Lấy danh sách Top 5 từ XGBoost Cache...")
    cache_file = DATA_DIR / "features_cache_5f.pkl"
    with open(cache_file, 'rb') as f:
        X_train, y_train, qids, test_processed = pickle.load(f)

    # Dùng XGBoost để mồi (Với thông số vàng)
    ranker = xgb.XGBRanker(n_jobs=1, tree_method="hist", n_estimators=150, learning_rate=0.05, max_depth=6)
    ranker.fit(X_train, y_train, qid=qids)

    TOP_K = 5
    xgb_top_results = []
    for item in test_processed:
        f_scores = ranker.predict(item["X"])
        sorted_idx = np.argsort(f_scores)[::-1][:TOP_K]
        xgb_top_results.append({
            "gt": item["gt"], 
            "ids": [item["map"][i] for i in sorted_idx], 
            "scores": [f_scores[i] for i in sorted_idx]
        })

    print("\n🚀 BƯỚC 2: Nạp Thẩm phán Qwen3-8B (Chế độ Few-shot CoT)...")
    model_name = "Qwen/Qwen3-8B" # Giữ nguyên cấu hình chuẩn của bạn
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    retriever = MultiViewHybridRetriever(load_models=False)
    chunk_dict = {f"{c['law_id']}_{c['article_id']}": c['text'] for c in retriever.chunks}

    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_raw_data = json.load(f)

    final_predictions = []
    qwen_logs = []

    print("\n🧠 BƯỚC 3: Bắt đầu Reranking và Ghi log lập luận...")
    for idx, raw_item in enumerate(tqdm(test_raw_data, desc="Qwen Evaluating")):
        query = raw_item.get("text", "")
        xgb_result = xgb_top_results[idx]
        gt_ids = xgb_result["gt"]
        
        candidates_scored = []
        item_log = {
            "query_id": idx,
            "query": query,
            "ground_truths": gt_ids,
            "evaluations": []
        }

        for art_key in xgb_result["ids"]:
            document_text = chunk_dict.get(art_key, "")
            if not document_text: continue

            messages = build_fewshot_cot_prompt(query, document_text)
            text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text_prompt], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                # Tăng max_new_tokens để Qwen có đủ không gian "nói nhảm" phần lập luận trước khi chốt
                outputs = model.generate(**inputs, max_new_tokens=250, do_sample=False)
            
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            final_score = 1.0 if "RELEVANCE_SCORE: 1" in response else 0.0
            
            candidates_scored.append({"art_key": art_key, "score": final_score})
            item_log["evaluations"].append({
                "article_id": art_key,
                "is_ground_truth": art_key in gt_ids,
                "qwen_response": response.strip(),
                "extracted_score": final_score
            })

        # Lọc ra các ứng viên được Qwen chấm 1.0
        sorted_candidates = [c["art_key"] for c in candidates_scored if c["score"] == 1.0]
        
        # Fallback: Nếu Qwen chê hết, lấy lại Top 1 của XGBoost
        if not sorted_candidates and xgb_result["ids"]:
            sorted_candidates = [xgb_result["ids"][0]]

        final_predictions.append({
            "gt_ids": gt_ids,
            "pred_ids": sorted_candidates
        })
        
        item_log["final_predicted"] = sorted_candidates
        qwen_logs.append(item_log)

    print("\n🔍 BƯỚC 4: Tính F2 và Lưu File Log...")
    f2_scores = [calculate_f2(p["gt_ids"], p["pred_ids"]) for p in final_predictions]
    
    log_file = DATA_DIR / "qwen_fewshot_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(qwen_logs, f, ensure_ascii=False, indent=4)

    print("\n" + "="*50)
    print("🔥 KẾT QUẢ: QWEN 3 (FEW-SHOT CoT) 🔥")
    print(f"F2-macro Đạt Được: {np.mean(f2_scores):.6f}")
    print(f"✅ Đã lưu toàn bộ nhật ký lập luận vào: {log_file}")
    print("="*50)

if __name__ == "__main__":
    run_qwen_fewshot()