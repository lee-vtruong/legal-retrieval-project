import pickle
from config import DATA_DIR

def main():
    path = f"{DATA_DIR}/features_cache_5f_with_text.pkl"
    print(f"🔍 Đang nội soi file: {path}")
    
    with open(path, 'rb') as f:
        data = pickle.load(f)

    print("Kiểu dữ liệu gốc:", type(data))
    
    if isinstance(data, dict):
        print("Các keys bên trong:")
        for k, v in data.items():
            shape_info = getattr(v, 'shape', len(v) if isinstance(v, (list, tuple)) else 'N/A')
            print(f" - Key: '{k}' | Type: {type(v)} | Shape/Len: {shape_info}")
            
    elif isinstance(data, (list, tuple)):
        print(f"Đây là Tuple/List có độ dài: {len(data)}")
        for i, item in enumerate(data):
            shape_info = getattr(item, 'shape', len(item) if isinstance(item, (list, tuple)) else 'N/A')
            print(f" - Item {i} | Type: {type(item)} | Shape/Len: {shape_info}")
            
    else:
        print("Shape:", getattr(data, 'shape', 'N/A'))

if __name__ == "__main__":
    main()