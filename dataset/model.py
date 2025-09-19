import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# ------------------------------
# 1️⃣ Load the graph
# ------------------------------
graph = torch.load("graph.pth")
print(graph)

# Node features and labels
# x = node feature (we only have name ID, but we can treat it as input)
# y = operator class (same as x in this simple example)
x = graph.x  # [num_nodes, 1]
y = x.squeeze()  # node-level labels = operator ID
num_classes = int(y.max().item()) + 1
x = F.one_hot(x.squeeze(), num_classes=num_classes).float()  # one-hot encoding

# ------------------------------
# 2️⃣ Train/test split
# ------------------------------
num_nodes = x.size(0)
perm = torch.randperm(num_nodes)
train_idx = perm[:int(0.8*num_nodes)]
test_idx = perm[int(0.8*num_nodes):]

# ------------------------------
# 3️⃣ Define a simple GCN
# ------------------------------
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GCN(in_channels=num_classes, hidden_channels=64, out_channels=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# ------------------------------
# 4️⃣ Training loop
# ------------------------------
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    out = model(x, graph.edge_index)
    loss = criterion(out[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()
    
    # Evaluate
    model.eval()
    pred = out.argmax(dim=1)
    acc = (pred[test_idx] == y[test_idx]).sum().item() / len(test_idx)
    if epoch % 5 == 0:
        print(f"Epoch {epoch:02d}, Loss: {loss.item():.4f}, Test Acc: {acc:.4f}")

# ------------------------------
# 5️⃣ Final prediction
# ------------------------------
model.eval()
pred = model(x, graph.edge_index)
pred_classes = pred.argmax(dim=1)
print("Predicted operator IDs for first 10 nodes:", pred_classes[:10])
