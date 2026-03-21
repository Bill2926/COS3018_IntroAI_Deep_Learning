# 🧠 ANN with PyTorch — The Titanic Cheat Sheet
> *Assumes you already have clean, scaled features ready to go.*

---

## Step 1 — Define Your Model Architecture

Build a class that inherits from `nn.Module`. Think of it as **designing the brain**: how many layers, how many neurons per layer.

```python
class TitanicModel(nn.Module):
    def __init__(self, in_features=10, h1=12, h2=6, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)   # Input → Hidden layer 1
        self.fc2 = nn.Linear(h1, h2)             # Hidden layer 1 → Hidden layer 2
        self.out = nn.Linear(h2, out_features)   # Hidden layer 2 → Output

    def forward(self, x):
        x = F.relu(self.fc1(x))   # Activation keeps only positive signals
        x = F.relu(self.fc2(x))
        x = self.out(x)           # Raw scores (logits), no activation here
        return x
```

**Key concepts:**
- `nn.Linear(a, b)` — a fully connected layer from `a` neurons to `b` neurons
- `F.relu` — activation function; kills negative values, lets positives through
- `out_features=2` — two output neurons because we have 2 classes (Survived / Died)

---

## Step 2 — Convert Data to PyTorch Tensors

PyTorch doesn't speak pandas — you need to convert everything to **tensors** (think: NumPy arrays, but GPU-compatible).

```python
X_train_t = torch.FloatTensor(X_train.values)   # Features → float
X_test_t  = torch.FloatTensor(X_test.values)
y_train_t = torch.LongTensor(y_train.values)     # Labels → int (class indices)
y_test_t  = torch.LongTensor(y_test.values)
```

> ⚠️ **Why Float vs Long?** Features are continuous numbers → `Float`. Labels are class indices (0 or 1) → `Long` (required by CrossEntropyLoss).

---

## Step 3 — Set Up Loss Function & Optimizer

```python
criterion = nn.CrossEntropyLoss()                          # Measures how wrong the model is
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Updates weights to reduce loss
```

**Key concepts:**
- **Loss function** = the score card. Lower is better.
- **CrossEntropyLoss** = standard choice for classification tasks
- **Adam optimizer** = a smart gradient descent that adapts the learning rate automatically
- **`lr` (learning rate)** = how big each update step is. Too high → overshoots. Too low → slow.

---

## Step 4 — The Training Loop

This is the **heart** of training. Repeat for N epochs:

```python
epochs = 100
train_losses, test_losses = [], []

for i in range(epochs):

    # --- TRAINING PHASE ---
    model.train()                          # Tell model: "we're training"
    y_pred = model(X_train_t)              # Forward pass: get predictions
    loss = criterion(y_pred, y_train_t)    # Calculate loss
    train_losses.append(loss.item())

    optimizer.zero_grad()   # 1. Clear old gradients
    loss.backward()         # 2. Backprop: compute new gradients
    optimizer.step()        # 3. Update weights

    # --- VALIDATION PHASE ---
    model.eval()                           # Tell model: "we're evaluating"
    with torch.no_grad():                  # Don't track gradients (saves memory)
        y_val = model(X_test_t)
        val_loss = criterion(y_val, y_test_t)
        test_losses.append(val_loss.item())
```

**The 3-step update ritual (never skip, never reorder):**
1. `zero_grad()` — reset gradients from last iteration
2. `backward()` — compute how much each weight contributed to the error
3. `step()` — nudge every weight in the right direction

> 🔍 **model.train() vs model.eval()** — Some layers (like Dropout, BatchNorm) behave differently during training vs evaluation. Always switch modes explicitly.

---

## Step 5 — Evaluate the Model

```python
model.eval()
with torch.no_grad():
    outputs = model(X_test_t)
    _, predicted = torch.max(outputs, 1)   # Pick the class with highest score

    correct = (predicted == y_test_t).sum().item()
    accuracy = correct / y_test_t.size(0) * 100

print(f'Accuracy: {accuracy:.2f}%')  # → 80.45%
```

- `torch.max(outputs, 1)` returns `(values, indices)` — we only care about `indices` (the predicted class)

---

## Step 6 — Single Prediction

```python
new_passenger = pd.DataFrame([{ ...features... }])
new_scaled = torch.tensor(scaler.transform(new_passenger)).float()

with torch.no_grad():
    pred_raw = model(new_scaled)
    prob = F.softmax(pred_raw, dim=1)          # Convert raw scores → probabilities (sum to 1)
    status = "Survived" if torch.argmax(prob) == 1 else "Died"
    print(f"{status} ({prob.max().item()*100:.2f}% confidence)")
```

> `softmax` squashes raw logits into a probability distribution. Use it only at inference, not during training — `CrossEntropyLoss` handles this internally.

---

## Step 7 — Save the Model

```python
torch.save(model.state_dict(), "titanic_model.pt")
```

`state_dict()` saves just the **weights**, not the architecture. To reload, you'd re-define the class and call `model.load_state_dict(torch.load(...))`.

---

## 🗺️ The Full Pipeline at a Glance

```
Clean Data (done)
      ↓
Define Model Class (architecture)
      ↓
Convert to Tensors
      ↓
Set Loss + Optimizer
      ↓
Training Loop (N epochs):
    Forward → Loss → zero_grad → backward → step
    Validate (no_grad)
      ↓
Evaluate Accuracy
      ↓
Single Predictions with softmax
      ↓
Save weights
```

---

## 📌 Quick Reference Card

| Thing | What it does |
|---|---|
| `nn.Linear(a, b)` | Fully connected layer, a → b neurons |
| `F.relu` | Activation: keeps positives, zeros out negatives |
| `CrossEntropyLoss` | Loss for classification (handles softmax internally) |
| `Adam(lr=0.01)` | Optimizer that updates weights each step |
| `zero_grad()` | Clears gradients before each backward pass |
| `loss.backward()` | Computes gradients via backpropagation |
| `optimizer.step()` | Applies gradients to update weights |
| `model.train()` | Training mode (activates Dropout etc.) |
| `model.eval()` | Evaluation mode |
| `torch.no_grad()` | Disables gradient tracking (faster, less memory) |
| `F.softmax` | Converts logits → probabilities at inference |
| `state_dict()` | Dictionary of all learned weights |