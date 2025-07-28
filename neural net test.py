import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle

with open ('dataset.txt', 'rb') as fp:
    dataset = pickle.load(fp)

with open ('labels.txt', 'rb') as fp:
    labels = pickle.load(fp)

class SlopeDataset(Dataset):
    def __init__(self):
        self.data = torch.stack([torch.tensor(i, dtype=torch.float32).unsqueeze(0) for i in dataset])
        self.labels = torch.tensor(labels, dtype=torch.long)  # 0=up, 1=down

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

# 2) Model definition
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PixelConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # pixel-wise linear layer: one neuron per pixel
        #self.pixel_fc = nn.Conv2d(1, 1, kernel_size=1)
        # convolutional layer
        #self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        # post-conv pixel-wise layer: one neuron per feature map pixel
        #self.post_conv_fc = nn.Conv2d(1, 1, kernel_size=1)
        # classifier: two outputs for up/down
        #self.classifier = nn.Linear(4*4*1, 2)

        self.quad_conv = nn.Conv2d(in_channels=1, out_channels=1,
                                   kernel_size=2, stride=2, padding=0)
        self.classifier = nn.Linear(2*2*1, 2)

    def forward(self, x):
        # x: [B,1,4,4]
        #x = self.pixel_fc(x)        # -> [B,1,4,4]
        #x = self.conv(x)            # -> [B,4,4,4]
        #x = self.post_conv_fc(x)    # -> [B,1,4,4]
        #x = x.view(x.size(0), -1)   # -> [B,16]
        #x = self.classifier(x)      # -> [B,2]

        x = self.quad_conv(x)        # -> [B,1,2,2]
        x = x.view(x.size(0), -1)   # -> [B,4]
        x = self.classifier(x)      # -> [B,2]
        return x

class PixelConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Vertical conv: process each column (height = 4, width = 1)
        self.vert_conv = nn.Conv2d(in_channels=1, out_channels=1,
                                   kernel_size=(4, 1), stride=1, padding=0)
        
        # Horizontal conv: process each row (height = 1, width = 4)
        self.horiz_conv = nn.Conv2d(in_channels=1, out_channels=1,
                                    kernel_size=(1, 4), stride=1, padding=0)

        # Output of vert_conv is [B, 1, 1, 4], horiz_conv is [B, 1, 4, 1]
        # Flatten and concatenate -> total 8 features
        self.classifier = nn.Linear(8, 2)

    def forward(self, x):
        # x shape: [B, 1, 4, 4]
        vert = self.vert_conv(x)      # -> [B, 1, 1, 4]
        horiz = self.horiz_conv(x)    # -> [B, 1, 4, 1]

        vert = vert.view(x.size(0), -1)    # -> [B, 4]
        horiz = horiz.view(x.size(0), -1)  # -> [B, 4]

        x = torch.cat([vert, horiz], dim=1)  # -> [B, 8]
        x = self.classifier(x)               # -> [B, 2]
        return x

def predict(matrix):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        return probs

class MiniSlopeViT(nn.Module):
    def __init__(self):
        super().__init__()
        # 1) Patch embedding: 2×2 patches → 4 patches, embed dim D=4
        #    - Linear projection: (2*2*1)=4 → 4
        self.patch_proj = nn.Linear(4, 4, bias=True)
        #    - +1 class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 4))
        #    - positional embeddings for 5 tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, 5, 4))

        # 2) Single Transformer block
        self.ln1 = nn.LayerNorm(4)
        #    1‑head self‑attention
        self.qkv = nn.Linear(4, 3*4, bias=True)
        self.attn_out = nn.Linear(4, 4, bias=True)
        self.ln2 = nn.LayerNorm(4)
        #    MLP: hidden 2×D=8
        self.mlp_fc1 = nn.Linear(4, 8, bias=True)
        self.mlp_fc2 = nn.Linear(8, 4, bias=True)

        # 3) Classification head: D→3 classes
        self.head = nn.Linear(4, 3, bias=True)

    def forward(self, x):
        # x: [B,1,4,4] →
        #  extract non‑overlapping 2×2 patches, flatten
        B = x.size(0)
        patches = x.unfold(2,2,2).unfold(3,2,2)  # → [B,1,2,2,2,2]
        patches = patches.contiguous().view(B, 4, 4)  # 4 patches of 4 dims
        x = self.patch_proj(patches)                  # [B,4,4]

        cls = self.cls_token.expand(B, -1, -1)        # [B,1,4]
        x = torch.cat([cls, x], dim=1)               # [B,5,4]
        x = x + self.pos_embed

        # Transformer block
        y = self.ln1(x)
        qkv = self.qkv(y).reshape(B, 5, 3, 4).permute(2,0,1,3)
        q,k,v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2,-1)) / (4**0.5)
        attn = scores.softmax(-1)
        y = (attn @ v)
        y = y.transpose(1,2).reshape(B,5,4)
        y = self.attn_out(y)
        x = x + y

        y = self.ln2(x)
        y = self.mlp_fc2(self.mlp_fc1(y).relu())
        x = x + y

        # classification from cls token
        cls_final = x[:,0]
        return self.head(cls_final)

# Instantiate dataset, dataloader, model, loss, optimizer
dataset = SlopeDataset()
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
model = MiniSlopeViT().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3) Training loop
model.train()
for epoch in range(1000):
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

# Test patterns
test_lst = []
test_lst.append(
    [[0,0,0,1],
    [0,0,1,0],
    [0,1,0,0],
    [1,0,0,0],]
)
test_lst.append(
    [[1,1,1,1],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],]
)
test_lst.append(
    [[0,0,0,0],
    [1,1,1,1],
    [0,0,0,0],
    [0,0,0,0],]
)
test_lst.append(
    [[0,0,0,0],
    [0,0,0,0],
    [1,1,1,1],
    [0,0,0,0],]
)
test_lst.append(
    [[0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [1,1,1,1],]
)
test_lst.append(
    [[1,0,0,0],
    [1,0,0,0],
    [1,0,0,0],
    [1,0,0,0],]
)
test_lst.append(
    [[0,1,0,0],
    [0,1,0,0],
    [0,1,0,0],
    [0,1,0,0],]
)
test_lst.append(
    [[0,0,1,0],
    [0,0,1,0],
    [0,0,1,0],
    [0,0,1,0],]
)
test_lst.append(
    [[0,0,0,1],
    [0,0,0,1],
    [0,0,0,1],
    [0,0,0,1],]
)
test_lst.append(
    [[1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1],]
)
test_lst = [predict(i) for i in test_lst]
[print(f"Up pattern -> P(up)={i[0]:.4f}, P(down)={i[1]:.4f}") for i in test_lst]

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)


