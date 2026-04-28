import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_prep.builder import CorpusBuilder

if __name__ == "__main__":
    print("="*50)
    print("BẮT ĐẦU QUÁ TRÌNH CHUẨN BỊ DỮ LIỆU")
    print("="*50)
    
    builder = CorpusBuilder()
    builder.run()