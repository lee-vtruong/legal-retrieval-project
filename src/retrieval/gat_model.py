import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class LegalGAT(torch.nn.Module):
    def __init__(self, in_channels=1024, hidden_channels=512, out_channels=1024):
        super(LegalGAT, self).__init__()
        # Kiến trúc 2-layer GAT theo đúng mô tả trong bài báo
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=False)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        # Lớp 1 + ReLU
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        # Lớp 2 (Vector đầu ra sẽ có cùng số chiều với BGE-M3 để dễ fusion)
        x = self.conv2(x, edge_index)
        return x

def contrastive_link_prediction_loss(z, edge_index):
    """
    Hàm Loss huấn luyện GAT không giám sát: 
    Các node có kết nối (edges) sẽ có vector gần nhau, không kết nối sẽ đẩy ra xa.
    """
    src, dst = edge_index
    # Positive pairs: similarity của các node có nối cạnh
    pos_sim = torch.sum(z[src] * z[dst], dim=-1)
    
    # Negative pairs: lấy ngẫu nhiên các node không nối với nhau
    neg_dst = torch.randint(0, z.size(0), (src.size(0),), device=z.device)
    neg_sim = torch.sum(z[src] * z[neg_dst], dim=-1)
    
    # Margin Ranking Loss
    loss = torch.mean(torch.relu(1.0 - pos_sim + neg_sim))
    return loss