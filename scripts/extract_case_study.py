import json
import pickle
import pandas as pd
from config import DATA_DIR

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    target_query = "Cơ sở sản xuất, mua bán sản phẩm giống vật nuôi không có quyền sau đây"
    print(f"🔍 ĐANG TRUY VẾT ĐIỂM SỐ CHO CÂU HỎI:\n'{target_query}'\n" + "="*80)

    try:
        # 1. Load file chứa ứng viên (Top-30) của tập Test
        candidates_data = load_json(f"{DATA_DIR}/test_candidates.json")
        
        # 2. Load file chứa điểm LLM của tập Test
        llm_data = load_pickle(f"{DATA_DIR}/test_sllm_feature.pkl")
        
        # 3. Load file log lỗi (nếu có để xem dự đoán cuối cùng)
        try:
            error_log = load_json(f"{DATA_DIR}/results/error_analysis_report.json")
        except:
            error_log = []

    except Exception as e:
        print(f"⚠️ Lỗi đọc file: {e}")
        return

    # --- TÌM DATA CỦA CÂU HỎI TRONG test_candidates.json ---
    query_cands = None
    if isinstance(candidates_data, list):
        for item in candidates_data:
            text = item.get('query_text', item.get('question', ''))
            if target_query in text:
                query_cands = item
                break
    elif isinstance(candidates_data, dict):
        if target_query in candidates_data:
            query_cands = candidates_data[target_query]
            
    if not query_cands:
        print("❌ Không tìm thấy câu hỏi này trong test_candidates.json")
        return

    # Lấy thông tin cơ bản
    gts = query_cands.get('gt_ids', query_cands.get('relevant_articles', []))
    doc_ids = query_cands.get('candidates', query_cands.get('doc_ids', []))
    
    # --- TÌM DATA TRONG test_sllm_feature.pkl ---
    llm_scores = {}
    if isinstance(llm_data, dict):
        if target_query in llm_data:
            llm_scores = llm_data[target_query]
        else:
            # Tìm gần đúng
            for k in llm_data.keys():
                if target_query in str(k):
                    llm_scores = llm_data[k]
                    break

    # --- LẬP BẢNG THỐNG KÊ ---
    rows = []
    for i, doc in enumerate(doc_ids):
        is_gt = "✅" if doc in gts else "❌"
        
        # Lấy các điểm Phase 1 & 2 (Nếu file json có lưu)
        s_hybrid = query_cands['s_hybrid'][i] if 's_hybrid' in query_cands else 0.0
        s_gat = query_cands['s_gat'][i] if 's_gat' in query_cands else 0.0
        s_xgb = query_cands['s_xgb'][i] if 's_xgb' in query_cands else 0.0
        
        # Lấy điểm LLM
        s_llm = llm_scores.get(doc, 0.0)
        
        rows.append({
            "GT": is_gt,
            "Doc_ID": doc,
            "S_hybrid": round(s_hybrid, 4),
            "S_gat": round(s_gat, 4),
            "S_xgb": round(s_xgb, 4),
            "S_llm": round(s_llm, 4)
        })

    df = pd.DataFrame(rows)
    
    # Sắp xếp thử theo s_llm hoặc s_hybrid để xem Top
    if df['S_llm'].sum() > 0:
        df = df.sort_values(by="S_llm", ascending=False).head(8)
    else:
        df = df.sort_values(by="S_hybrid", ascending=False).head(8)
        
    print(df.to_string(index=False))
    
    # --- CHECK KẾT QUẢ CUỐI CÙNG TRONG ERROR LOG ---
    print("\n" + "="*80)
    for err in error_log:
        if target_query in err.get('query_text', ''):
            print(f"📌 DỰ ĐOÁN CUỐI CÙNG CỦA HỆ THỐNG: {err.get('predicted', [])}")
            break

if __name__ == "__main__":
    main()