import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_prep.vectorizer import VectorBuilder

if __name__ == "__main__":
    print("="*50)
    print("BẮT ĐẦU QUÁ TRÌNH TRÍCH XUẤT VECTOR & INDEX")
    print("="*50)
    
    builder = VectorBuilder()
    builder.run()