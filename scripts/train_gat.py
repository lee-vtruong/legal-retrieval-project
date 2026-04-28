import json
import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNK_FILE, DATA_DIR, EMBEDDINGS_DIR, DEVICE
from src.retrieval.gat_model import LegalGAT, contrastive_link_prediction_loss

def build_graph_data():
    print("1. Đang xây dựng cấu trúc Đồ thị (Graph)...")
    
    # Nạp Chunks để lập mapping Node ID
    with open(CHUNK_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    # Chỉ lấy các chunk ở level "article" để làm Node cho đồ thị
    article_chunks = [c for c in chunks if c["level"] == "article"]
    node_mapping = {f"{c['law_id']}_{c['article_id']}": idx for idx, c in enumerate(article_chunks)}
    print(f"Tổng số Nodes (Điều luật): {len(node_mapping)}")

    # Trích xuất vector BGE-M3 tương ứng với các Article làm initial features
    all_bge_emb = np.load(EMBEDDINGS_DIR / "bge_m3_embeddings.npy")
    article_indices = [idx for idx, c in enumerate(chunks) if c["level"] == "article"]
    node_features = torch.tensor(all_bge_emb[article_indices], dtype=torch.float32)

    # Nạp Edges
    with open(DATA_DIR / "graphs" / "citation_edges.json", 'r', encoding='utf-8') as f:
        edges_data = json.load(f)
        
    edge_list = []
    for edge in edges_data:
        src = edge["source"]
        tgt = edge["target"]
        if src in node_mapping and tgt in node_mapping:
            edge_list.append([node_mapping[src], node_mapping[tgt]])
            
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    print(f"Tổng số Edges (Liên kết trích dẫn hợp lệ): {edge_index.size(1)}")
    
    data = Data(x=node_features, edge_index=edge_index)
    return data, article_chunks

def train_and_extract_gat():
    data, article_chunks = build_graph_data()
    data = data.to(DEVICE)
    
    # Khởi tạo GAT Model
    model = LegalGAT(in_channels=1024, hidden_channels=512, out_channels=1024).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\n2. Bắt đầu huấn luyện GAT (Link Prediction)...")
    model.train()
    epochs = 200 # Huấn luyện đồ thị hội tụ khá nhanh
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        loss = contrastive_link_prediction_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} - Loss: {loss.item():.4f}")
            
    print("\n3. Trích xuất GAT Embeddings...")
    model.eval()
    with torch.no_grad():
        final_embeddings = model(data.x, data.edge_index).cpu().numpy()
        
    # Lưu metadata để khi truy xuất biết vector nào thuộc article nào
    np.save(EMBEDDINGS_DIR / "gat_embeddings.npy", final_embeddings)
    with open(EMBEDDINGS_DIR / "gat_node_mapping.json", "w", encoding="utf-8") as f:
        json.dump([{"law_id": c["law_id"], "article_id": c["article_id"]} for c in article_chunks], f, indent=2)
        
    print("✅ Đã lưu trọng số và vector GAT thành công!")

if __name__ == "__main__":
    train_and_extract_gat()