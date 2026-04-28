import json, os, pickle, re
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Ép dùng GPU 5 cho LLM
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Chuyển từ 5 sang 0
os.environ['VLLM_USE_V1'] = '0' 

def extract_scores_robust(text):
    """Tìm điểm số 0-10 trong chuỗi JSON hoặc text thô."""
    score_map = {}
    # Tìm tất cả các cặp "ID": Điểm
    matches = re.findall(r'"([^"]+)":\s*(\d+\.?\d*)', text)
    for art_id, score in matches:
        try:
            score_map[art_id.strip()] = float(score) / 10.0 # Normalize 0-1
        except:
            continue
    return score_map

def run_scoring():
    print("🤖 Đang khởi tạo Qwen3-14B trên GPU 5...")
    llm = LLM(model="Qwen/Qwen3-14B", trust_remote_code=True, gpu_memory_utilization=0.45)
    samp = SamplingParams(temperature=0.0, max_tokens=1024) # Tăng max_tokens để chấm được nhiều bài cùng lúc

    with open("data/test_candidates.json", "r", encoding='utf-8') as f:
        data = json.load(f)
    
    final_scores_cache = []

    for item in tqdm(data, desc="LLM Scoring"):
        query = item["query"]
        candidates = item["candidates"] # List of {"id": ..., "text": ...}
        
        cand_str = "\n".join([f"{i+1}. {c['id']}: {c['text'][:300]}" for i, c in enumerate(candidates)])
        
        prompt = f"""<|im_start|>system
Bạn là trợ lý pháp lý cao cấp. Hãy chấm điểm mức độ liên quan của các điều luật với câu hỏi (thang điểm 0-10).
Yêu cầu: Chỉ trả về JSON duy nhất với key "scores".
Ví dụ: {{"scores": {{"Luật_1": 9.5, "Luật_2": 2.0}}}}
<|im_end|>
<|im_start|>user
Câu hỏi: {query}
Ứng viên:
{cand_str}
<|im_end|>
<|im_start|>assistant
Lập luận: """ # Để nó lập luận một chút rồi mới ra JSON sẽ chính xác hơn

        try:
            outputs = llm.generate([prompt], samp, use_tqdm=False)
            raw_text = outputs[0].outputs[0].text
            score_map = extract_scores_robust(raw_text)
        except Exception as e:
            print(f"Error at query: {query[:30]} - {e}")
            score_map = {}

        # Đảm bảo mỗi candidate đều có điểm (fallback 0.5 nếu LLM quên)
        for c in candidates:
            if c['id'] not in score_map:
                score_map[c['id']] = 0.5
        
        final_scores_cache.append(score_map)

    with open("data/test_sllm_feature.pkl", "wb") as f:
        pickle.dump(final_scores_cache, f)
    print(f"\n✅ Đã lưu Feature thứ 6 vào data/test_sllm_feature.pkl")

if __name__ == "__main__":
    run_scoring()