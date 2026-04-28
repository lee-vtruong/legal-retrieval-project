import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval.hybrid import MultiViewHybridRetriever

if __name__ == "__main__":
    retriever = MultiViewHybridRetriever()
    query = "Hành vi sử dụng trái phép chất ma túy sẽ bị xử phạt như thế nào?"
    
    print(f"\n🔍 Truy vấn: {query}")
    results = retriever.retrieve(query, top_k=3)
    
    for i, res in enumerate(results):
        print(f"\nTop {i+1} [Điểm: {res['hybrid_score']:.4f}] - Level: {res['level']}")
        print(f"Luật: {res['law_id']} - Điều: {res['article_id']}")
        print(f"Trích xuất: {res['text'][:150]}...")