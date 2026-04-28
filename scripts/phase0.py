import json, os, pickle
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval.hybrid import MultiViewHybridRetriever
from config import DATA_DIR, TEST_FILE

def prepare_candidates():
    print("📚 Nạp Hybrid Retriever để trích xuất ứng viên...")
    retriever = MultiViewHybridRetriever(load_models=True)
    
    with open(TEST_FILE, 'r') as f:
        test_data = json.load(f)
    
    # Chỉ chạy trên Test Set để tối ưu thời gian (hoặc Train nếu bạn muốn)
    candidates_data = []
    
    for item in tqdm(test_data, desc="Extracting Candidates"):
        query = item["text"]
        # Lấy Top 15 từ Hybrid để đảm bảo Recall
        scores = retriever._get_combined_scores(query)
        top_indices = np.argsort(scores)[::-1][:15]
        
        candidates = []
        seen_ids = set()
        for idx in top_indices:
            chunk = retriever.chunks[idx]
            art_id = f"{chunk['law_id']}_{chunk['article_id']}"
            if art_id not in seen_ids:
                seen_ids.add(art_id)
                candidates.append({"id": art_id, "text": chunk['text']})
        
        candidates_data.append({
            "query": query,
            "gt": [f"{r['law_id']}_{r['article_id']}" for r in item.get("relevant_articles", [])],
            "candidates": candidates
        })

    os.makedirs("data", exist_ok=True)
    with open("data/test_candidates.json", "w", encoding='utf-8') as f:
        json.dump(candidates_data, f, ensure_ascii=False, indent=2)
    print("✅ Đã tạo xong file data/test_candidates.json!")

if __name__ == "__main__":
    prepare_candidates()