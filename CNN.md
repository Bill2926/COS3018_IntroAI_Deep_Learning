# 🖼️ CNN with PyTorch — The MNIST Cheat Sheet
> *For when you've got your data ready and just need to know how CNNs work.*

---

## How CNNs Differ from ANNs

In ANNs, you flatten everything into a 1D vector. CNNs instead slide **filters** over the image — learning spatial patterns like edges, curves, and shapes — before finally passing to fully connected layers for classification.

```
Image (2D) → Conv → Pool → Conv → Pool → Flatten → FC Layers → Output
```

---

## Step 1 — Load Data with torchvision

Unlike tabular data, images come with their own PyTorch tooling.

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Convert images to tensors (normalizes pixel values to [0, 1])
transform = transforms.ToTensor()

train_data = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_data  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Wrap in DataLoader: batches + shuffling
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=10, shuffle=True)
```

**Key concepts:**
- `transforms.ToTensor()` — converts PIL images to tensors with shape `[C, H, W]`
- `DataLoader` — feeds data in **batches** during training, much more efficient than one image at a time
- MNIST images are `[1, 28, 28]` — 1 channel (grayscale), 28×28 pixels

---

## Step 2 — Understand What Conv + Pool Does to Shape

Before writing the model, it's useful to trace how image dimensions change through layers.

```python
# Starting shape: [1, 28, 28]  (batch=1, channels=1, H=28, W=28)

x = F.relu(conv1(x))      # conv1: in_ch=1, out_ch=6, kernel=3, stride=1
# Shape: [1, 6, 26, 26]   → (28-3)/1 + 1 = 26

x = F.max_pool2d(x, 2, 2) # 2x2 pool, stride=2
# Shape: [1, 6, 13, 13]   → 26/2 = 13

x = F.relu(conv2(x))      # conv2: in_ch=6, out_ch=16, kernel=3, stride=1
# Shape: [1, 16, 11, 11]  → (13-3)/1 + 1 = 11

x = F.max_pool2d(x, 2, 2) # 2x2 pool, stride=2
# Shape: [1, 16, 5, 5]    → 11/2 = 5 (floor)
```

> 🧮 **Shape formula after Conv:** `floor((input - kernel) / stride) + 1`
>
> 🧮 **Shape formula after MaxPool:** `floor(input / kernel)` (when stride = kernel)

This is why `fc1` takes `16*5*5 = 400` as its input size — you need to work out the math before you define FC layers!

---

## Step 3 — Define the CNN Model

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, 3, 1)    # 1 in-channel, 6 filters, 3x3 kernel, stride 1
        self.conv2 = nn.Conv2d(6, 16, 3, 1)   # 6 in-channels, 16 filters, 3x3 kernel, stride 1
        # Fully connected layers (ANN head)
        self.fc1 = nn.Linear(16*5*5, 120)      # 400 → 120
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)           # 10 output classes (digits 0-9)

    def forward(self, X):
        # Conv block 1
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)

        # Conv block 2
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)

        # Flatten: [batch, 16, 5, 5] → [batch, 400]
        X = X.view(-1, 16*5*5)

        # FC layers
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim=1)        # log probabilities for each class
```

**Key concepts:**
- `nn.Conv2d(in_ch, out_ch, kernel, stride)` — the convolutional layer
- `F.max_pool2d(x, kernel, stride)` — downsamples by taking the max in each window
- `x.view(-1, 16*5*5)` — flattens the 3D feature map into 1D for the FC layers
- `F.log_softmax` — outputs **log probabilities** (pairs with `nn.CrossEntropyLoss` or `nn.NLLLoss`)

> ⚠️ `out_features` of the **last FC layer = number of classes** (10 for digits 0-9)

---

## Step 4 — Loss & Optimizer

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
```

Same as ANN — nothing changes here.

---

## Step 5 — Training Loop (with DataLoader)

The key difference from the ANN version: data now comes in **batches** from `DataLoader`, so you loop over batches inside each epoch.

```python
epochs = 5
train_losses, test_losses = [], []
train_correct, test_correct = [], []

for i in range(epochs):
    train_corr = 0
    test_corr  = 0

    # --- TRAINING PHASE ---
    for batch, (X_train, y_train) in enumerate(train_loader):
        y_pred = cnn(X_train)                              # Forward pass
        loss = criterion(y_pred, y_train)                  # Calculate loss

        predicted = torch.max(y_pred.data, 1)[1]           # Predicted class indices
        train_corr += (predicted == y_train).sum()         # Count correct

        optimizer.zero_grad()   # Clear gradients
        loss.backward()         # Backprop
        optimizer.step()        # Update weights

    train_losses.append(loss)
    train_correct.append(train_corr)

    # --- VALIDATION PHASE ---
    with torch.no_grad():
        for batch, (X_test, y_test) in enumerate(test_loader):
            y_val = cnn(X_test)
            predicted = torch.max(y_val.data, 1)[1]
            test_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(test_corr)
```

**Why `torch.max(y_pred.data, 1)[1]`?**
- `y_pred` has shape `[batch_size, 10]` — one score per class
- `torch.max(..., 1)` finds the max along dimension 1 (across classes)
- `[1]` gives the **index** (the predicted class), not the value

---

## Step 6 — Plot Loss & Accuracy

```python
# Loss curve
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses,  label="Validation Loss")
plt.title("Loss per Epoch")
plt.legend()

# Accuracy curve
plt.plot([t/600 for t in train_correct], label="Training Accuracy")   # 60,000 / batch_size=10 → 6000 batches, but /600 ≈ percent
plt.plot([t/100 for t in test_correct], label="Validation Accuracy")
plt.title("Accuracy per Epoch")
plt.legend()
```

---

## 🗺️ CNN vs ANN: The Key Differences

| | ANN (Titanic) | CNN (MNIST) |
|---|---|---|
| Input | 1D feature vector | 2D image `[C, H, W]` |
| Data loading | pandas + manual tensors | `torchvision.datasets` + `DataLoader` |
| Model layers | Only FC layers | Conv → Pool → Flatten → FC |
| Training loop | One pass per epoch | Batch loop inside each epoch |
| Output activation | `softmax` at inference | `log_softmax` in forward pass |

---

## 📌 Quick Reference Card

| Thing | What it does |
|---|---|
| `transforms.ToTensor()` | Converts image to tensor, normalizes pixels to [0,1] |
| `DataLoader(data, batch_size, shuffle)` | Wraps dataset for batched training |
| `nn.Conv2d(in, out, kernel, stride)` | Learns spatial filters over the image |
| `F.max_pool2d(x, kernel, stride)` | Downsamples by keeping max in each window |
| `x.view(-1, N)` | Flattens 3D feature map to 1D for FC layers |
| `F.log_softmax(x, dim=1)` | Log probabilities across classes (use with NLLLoss) |
| `torch.max(x, 1)[1]` | Index of the highest-scoring class (predicted label) |
| `(predicted == y).sum()` | Counts how many predictions were correct |