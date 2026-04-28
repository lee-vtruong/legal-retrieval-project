import json
import os
import sys
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR
from src.retrieval.hybrid import MultiViewHybridRetriever

def extract_json_array(text):
    """Hàm tìm mảng JSON ở cuối câu trả lời của LLM"""
    try:
        # Tìm tất cả các cụm có ngoặc vuông
        matches = re.findall(r'\[.*?\]', text, re.DOTALL)
        if matches:
            # Lấy cái ngoặc vuông xuất hiện cuối cùng (là đáp án chốt)
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

def run_llm_reranker():
    print("🚀 Đang nạp Dữ liệu văn bản luật...")
    retriever = MultiViewHybridRetriever(load_models=False) 
    article_texts = {}
    for chunk in retriever.chunks:
        key = f"{chunk['law_id']}_{chunk['article_id']}"
        if key not in article_texts:
            article_texts[key] = ""
        article_texts[key] += chunk['text'] + "\n"

    print("📂 Đọc kết quả Top 30 từ GAM-Hybrid...")
    submission_file = DATA_DIR / "results" / "final_submission.json"
    with open(submission_file, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)

    # ==========================================
    # 🔍 ĐO LƯỜNG RECALL TRƯỚC KHI GỌI LLM
    # ==========================================
    hits_at_5 = 0
    hits_at_10 = 0
    valid_queries = 0

    for item in predictions_data:
        gt_ids = item.get("relevant_articles", [])
        if not gt_ids: continue
        valid_queries += 1
        
        # Trích xuất ID của Top 10 từ file JSON
        pred_ids = [f"{p['law_id']}_{p['article_id']}" for p in item.get("predictions", [])]
        
        if any(gt in pred_ids[:5] for gt in gt_ids): hits_at_5 += 1
        if any(gt in pred_ids[:10] for gt in gt_ids): hits_at_10 += 1

    print("\n" + "="*50)
    print("📊 TRẠNG THÁI DỮ LIỆU ĐẦU VÀO (LIMIT CEILING)")
    print(f"Recall@5  : {hits_at_5 / valid_queries:.4f} (Đây là giới hạn cũ của bạn)")
    print(f"Recall@10 : {hits_at_10 / valid_queries:.4f} (Mục tiêu mới của chúng ta)")
    print("="*50 + "\n")

    print("🧠 Khởi tạo vLLM Engine (Qwen2.5-7B-Instruct)...")
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True, gpu_memory_utilization=0.6)
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=1024)

    prompts = []
    ground_truths = []
    
    # 🔥 ĐÃ NÂNG LÊN TOP 10 ĐỂ BẮT HẾT ĐÁP ÁN
    TOP_K = 10 
    
    for item in predictions_data:
        query = item["text"]
        gt_ids = item.get("relevant_articles", [])
        if not gt_ids: continue # Bỏ qua câu không có nhãn
        ground_truths.append(gt_ids)
        
        preds = item.get("predictions", [])[:TOP_K]
        
        # 1. Ghép nối nội dung Top K
        context_str = ""
        for i, p in enumerate(preds):
            art_id = f"{p['law_id']}_{p['article_id']}"
            text = article_texts.get(art_id, "Nội dung không tồn tại.")
            # Cắt bớt xuống 300 từ mỗi văn bản để chứa đủ 10 văn bản
            text = " ".join(text.split()[:300]) 
            context_str += f"\n[Văn bản {i+1}] ID: {art_id}\nNội dung: {text}\n"

        # 2. Xây dựng Prompt "Tối giản"
        # 2. Xây dựng Few-Shot CoT Prompt
        # 2. Xây dựng Few-Shot CoT Prompt (Chuẩn format ALQAC)
        prompt = f"""<|im_start|>system
Bạn là Thẩm phán chuyên phân tích pháp lý.
Nhiệm vụ: Đọc CÂU HỎI và DANH SÁCH VĂN BẢN, tìm ra các văn bản TRỰC TIẾP trả lời câu hỏi.
LƯU Ý CỰC KỲ QUAN TRỌNG: 
1. Kết quả trả về phải copy CHÍNH XÁC chuỗi nằm ở phần "ID:". 
2. TUYỆT ĐỐI KHÔNG tự ý bỏ dấu tiếng Việt, KHÔNG thay đổi khoảng trắng thành dấu gạch dưới. Giữ nguyên 100% định dạng gốc.

Bạn PHẢI thực hiện 2 bước:
- Phân tích: Suy luận ngắn gọn lý do chọn/loại văn bản.
- Kết luận: In ra DUY NHẤT một mảng JSON chứa các ID đúng.
<|im_end|>
<|im_start|>user
CÂU HỎI: Cơ quan nào có thẩm quyền xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ?

DANH SÁCH VĂN BẢN:
[Văn bản 1] ID: Luật Giao thông đường bộ_11
Nội dung: Người tham gia giao thông phải chấp hành hiệu lệnh và chỉ dẫn của hệ thống báo hiệu đường bộ.
[Văn bản 2] ID: Nghị định 100/2019/NĐ-CP_74
Nội dung: Cảnh sát giao thông trong phạm vi chức năng, nhiệm vụ được giao có thẩm quyền xử phạt đối với các hành vi vi phạm quy định.
<|im_end|>
<|im_start|>assistant
Phân tích: Văn bản 1 chỉ quy định về nghĩa vụ chấp hành của người tham gia giao thông, không nhắc đến thẩm quyền xử phạt. Văn bản 2 quy định trực tiếp thẩm quyền xử phạt thuộc về Cảnh sát giao thông, trả lời đúng trọng tâm câu hỏi.
Kết luận: ["Nghị định 100/2019/NĐ-CP_74"]<|im_end|>
<|im_start|>user
CÂU HỎI: {query}

DANH SÁCH VĂN BẢN:
{context_str}
<|im_end|>
<|im_start|>assistant
Phân tích:"""
        
        prompts.append(prompt)

    print(f"🔥 Bắt đầu suy luận LLM cho {len(prompts)} câu hỏi...")
    outputs = llm.generate(prompts, sampling_params)

    f2_scores = []
    llm_predictions = []
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        pred_ids = extract_json_array(generated_text)
        gt_ids = ground_truths[i]
        
        # Gọt khoảng trắng bằng .strip()
        pred_ids = [str(pid).strip() for pid in pred_ids] if isinstance(pred_ids, list) else []
        
        f2 = calculate_f2(gt_ids, pred_ids)
        f2_scores.append(f2)
        
        llm_predictions.append({
            "query": predictions_data[i]["text"],
            "ground_truth": gt_ids,
            "llm_output_raw": generated_text,
            "llm_extracted": pred_ids,
            "f2_score": f2
        })

    final_f2_macro = np.mean([f for f in f2_scores if f is not None])
    
    out_json = DATA_DIR / "results" / "llm_listwise_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(llm_predictions, f, ensure_ascii=False, indent=4)

    print("\n" + "🌟"*25)
    print(f"KẾT QUẢ GAM-LLM HYBRID (STAGE 6 - LISTWISE RERANKING)")
    print(f"F2-macro Cuối Cùng: {final_f2_macro:.6f}")
    print("🌟"*25)
    print(f"✅ Đã lưu kết quả chi tiết tại: {out_json}")

if __name__ == "__main__":
    run_llm_reranker()