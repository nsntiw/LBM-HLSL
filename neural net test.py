import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1) Define dataset
matrix_lst = []
matrix_lst.append(
    [[0,0,0,1],
    [0,0,1,0],
    [0,1,0,0],
    [1,0,0,0],]
)
#--
matrix_lst.append(
    [[0,0,0,0],
    [0,0,0,0],
    [0,0,1,1],
    [1,1,0,0],]
)
matrix_lst.append(
    [[0,0,0,0],
    [0,0,1,1],
    [1,1,0,0],
    [0,0,0,0],]
)
matrix_lst.append(
    [[0,0,1,1],
    [1,1,0,0],
    [0,0,0,0],
    [0,0,0,0],]
)
#--
matrix_lst.append(
    [[0,0,0,0],
    [0,0,1,1],
    [0,1,1,0],
    [1,1,0,0],]
)
matrix_lst.append(
    [[0,0,1,1],
    [0,1,1,0],
    [1,1,0,0],
    [0,0,0,0],]
)
matrix_lst.append(
    [[0,0,0,0],
    [0,0,0,0],
    [0,0,0,1],
    [1,1,1,0],]
)
matrix_lst.append(
    [[0,0,0,0],
    [0,0,0,1],
    [1,1,1,0],
    [0,0,0,0],]
)
matrix_lst.append(
    [[0,0,0,1],
    [1,1,1,0],
    [0,0,0,0],
    [0,0,0,0],]
)
matrix_lst.append(
    [[0,1,1,1],
    [1,0,0,0],
    [0,0,0,0],
    [0,0,0,0],]
)
matrix_lst.append(
    [[0,0,0,0],
    [0,1,1,1],
    [1,0,0,0],
    [0,0,0,0],]
)
matrix_lst.append(
    [[0,0,0,0],
    [0,0,0,0],
    [0,1,1,1],
    [1,0,0,0],]
)
#==================
matrix_lst.append(
    [[1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1],]
)
#--
matrix_lst.append(
    [[1,1,0,0],
    [0,0,1,1],
    [0,0,0,0],
    [0,0,0,0],]
)
matrix_lst.append(
    [[0,0,0,0],
    [0,0,0,0],
    [1,1,0,0],
    [0,0,1,1],]
)
matrix_lst.append(
    [[0,0,0,0],
    [1,1,0,0],
    [0,0,1,1],
    [0,0,0,0],]
)
#--
matrix_lst.append(
    [[1,1,0,0],
    [0,1,1,0],
    [0,0,1,1],
    [0,0,0,0],]
)
matrix_lst.append(
    [[0,0,0,0],
    [1,1,0,0],
    [0,1,1,0],
    [0,0,1,1],]
)
matrix_lst.append(
    [[0,0,0,0],
    [0,0,0,0],
    [1,1,1,0],
    [0,0,0,1],]
)
matrix_lst.append(
    [[0,0,0,0],
    [1,1,1,0],
    [0,0,0,1],
    [0,0,0,0],]
)
matrix_lst.append(
    [[1,1,1,0],
    [0,0,0,1],
    [0,0,0,0],
    [0,0,0,0],]
)
matrix_lst.append(
    [[1,0,0,0],
    [0,1,1,1],
    [0,0,0,0],
    [0,0,0,0],]
)
matrix_lst.append(
    [[0,0,0,0],
    [1,0,0,0],
    [0,1,1,1],
    [0,0,0,0],]
)
matrix_lst.append(
    [[0,0,0,0],
    [0,0,0,0],
    [1,0,0,0],
    [0,1,1,1],]
)
#==================
matrix_lst.append(
    [[0,0,0,0],
    [0,0,0,0],
    [0,1,1,0],
    [1,0,0,1],]
)
matrix_lst.append(
    [[0,0,0,0],
    [0,1,1,0],
    [1,0,0,1],
    [0,0,0,0],]
)
matrix_lst.append(
    [[0,1,1,0],
    [1,0,0,1],
    [0,0,0,0],
    [0,0,0,0],]
)
matrix_lst.append(
    [[1,0,0,1],
    [0,1,1,0],
    [0,0,0,0],
    [0,0,0,0],]
)
matrix_lst.append(
    [[0,0,0,0],
    [1,0,0,1],
    [0,1,1,0],
    [0,0,0,0],]
)
matrix_lst.append(
    [[0,0,0,0],
    [0,0,0,0],
    [1,0,0,1],
    [0,1,1,0],]
)
label_lst=[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0.5,0.5,0.5,0.5,0.5,0.5]
class SlopeDataset(Dataset):
    def __init__(self):
        self.data = torch.stack([torch.tensor(i, dtype=torch.float32).unsqueeze(0) for i in matrix_lst])
        self.labels = torch.tensor(label_lst, dtype=torch.long)  # 0=up, 1=down

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

# Instantiate dataset, dataloader, model, loss, optimizer
dataset = SlopeDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
model = PixelConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3) Training loop
model.train()
for epoch in range(200):
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

# 4) Evaluation
def predict(matrix):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        return probs

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



    [a,0,0,0]    [0,0,0,0]
    [0,0,0,0]    [0,0,0,0]
    [0,0,0,0]    [0,0,0,0]
    [0,0,0,0]    [0,0,0,0]
