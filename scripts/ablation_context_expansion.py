#!/usr/bin/env python3
"""
Ablation Study Nhóm 3 (Run 10): Context Expansion (Vết dầu loang)
Thử nghiệm cộng thêm text của Điều trước và Điều sau để Rerank bằng Cross-Encoder.
"""

import os, sys, json, torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import CrossEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEST_FILE, MODELS_DIR
from src.retrieval.hybrid import MultiViewHybridRetriever

def calculate_f2(gt_ids, pred_ids):
    gts, preds = set(gt_ids), set(pred_ids)
    if not gts: return 0.0
    tp = len(preds & gts)
    fp = len(preds - gts)
    fn = len(gts - preds)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    return (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0

def run_context_expansion():
    print("📚 Đang nạp Retriever và Cross-Encoder...")
    retriever = MultiViewHybridRetriever(load_models=True)
    ce_model = CrossEncoder(str(MODELS_DIR / "fine_tuned_ce_alqac"), device="cuda:0")

    with open(TEST_FILE, 'r') as f:
        test_data = json.load(f)

    # Hàm lấy Vết dầu loang (cộng dồn text)
    def get_expanded_text(chunk_idx):
        curr = retriever.chunks[chunk_idx]
        law_id = curr['law_id']
        art_id = curr['article_id']
        
        # Tìm các chunk lân cận (giả định list chunks đã được sort theo luật và điều)
        text_prev = retriever.chunks[chunk_idx-1]['text'] if chunk_idx > 0 and retriever.chunks[chunk_idx-1]['law_id'] == law_id else ""
        text_next = retriever.chunks[chunk_idx+1]['text'] if chunk_idx < len(retriever.chunks)-1 and retriever.chunks[chunk_idx+1]['law_id'] == law_id else ""
        
        # Nối lại: [Prev] + [Curr] + [Next]
        expanded = f"{text_prev}\n{curr['text']}\n{text_next}".strip()
        # Cắt bớt nếu quá dài để tránh tràn VRAM của Cross-Encoder
        return expanded[:1500] 

    f2_list = []
    print("🚀 Đang chạy Rerank với Context Expansion (Top 30)...")
    
    for item in tqdm(test_data, desc="Context Expansion"):
        query = item["text"]
        gt_ids = [f"{r['law_id']}_{r['article_id']}" for r in item.get("relevant_articles", [])]
        
        # Lấy Top 30 từ Hybrid như cũ
        scores = retriever._get_combined_scores(query)
        top_indices = np.argsort(scores)[::-1][:30]
        
        # DÙNG EXPANDED TEXT THAY VÌ TEXT GỐC
        pairs = [[query, get_expanded_text(idx)] for idx in top_indices]
        
        # Chạy Cross-Encoder
        ce_scores = ce_model.predict(pairs, batch_size=4, convert_to_numpy=True)
        
        # Gắn map và tính F2
        d = {}
        for i, idx in enumerate(top_indices):
            art_key = f"{retriever.chunks[idx]['law_id']}_{retriever.chunks[idx]['article_id']}"
            if art_key not in d or ce_scores[i] > d[art_key]: 
                d[art_key] = ce_scores[i]
                
        art_list = sorted(d.items(), key=lambda x: x[1], reverse=True)
        top_s = art_list[0][1]
        
        # Margin chuẩn của CE thường là 1.0 (do dải điểm rộng hơn XGBoost)
        preds = [a for a, s in art_list if top_s - s <= 1.0] 
        f2_list.append(calculate_f2(gt_ids, preds))

    final_f2 = np.mean(f2_list)
    print("\n" + "="*50)
    print("🏆 KẾT QUẢ ABLATION: CONTEXT EXPANSION (Vết dầu loang) 🏆")
    print("="*50)
    print(f"Ngữ cảnh : [Điều trước] + [Điều hiện tại] + [Điều sau]")
    print(f"F2-Macro : {final_f2:.6f}")
    print("="*50)

if __name__ == "__main__":
    run_context_expansion()