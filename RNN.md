# 🔁 RNN with PyTorch — The IMDB Sentiment Cheat Sheet
> *For when you've got text data and need to classify sequences.*

---

## How RNNs Differ from ANNs and CNNs

ANNs and CNNs treat each input independently. RNNs are built for **sequences** — they process one token at a time and carry a **hidden state** forward, like short-term memory.

```
Word 1 → [RNN Cell] → hidden state h1
Word 2 → [RNN Cell] + h1 → hidden state h2
Word 3 → [RNN Cell] + h2 → hidden state h3
...
Last Word → [RNN Cell] + h(n-1) → final hidden state → FC → Output
```

> The key idea: the final hidden state is a **compressed summary** of the entire sequence.

---

## The Text Pipeline (What's New vs Images)

Text can't go into a neural network raw. You need three preprocessing steps unique to NLP:

```
Raw Text → Tokenize → Encode (vocab lookup) → Pad → Model
```

---

## Step 1 — Tokenize

Break each review into a list of words.

```python
df['text'] = df['text'].str.lower().str.split()
# "I loved this film" → ['i', 'loved', 'this', 'film']
```

---

## Step 2 — Build a Vocabulary & Encode

Map every unique word to an integer index. Think of it as a rulebook: word → number.

```python
vocab = set()
for phrase in df['text']:
    for word in phrase:
        vocab.add(word)

word_to_idx = {}
for idx, word in enumerate(vocab, start=1):   # start=1, because 0 is reserved for padding
    word_to_idx[word] = idx

# Result: {'the': 1, 'film': 2, 'loved': 3, ...}
```

> ⚠️ **Why start=1?** Index 0 is reserved for **padding tokens**. The model needs to distinguish "this word doesn't exist" from actual words.

---

## Step 3 — Pad Sequences

RNNs need fixed-length inputs for batching (matrix ops require equal sizes). Pad shorter sequences with zeros.

```python
max_length = df['text'].str.len().max()   # find the longest review

def encode_and_pad(text):
    encoded = [word_to_idx[word] for word in text]
    return encoded + [0] * (max_length - len(encoded))   # pad with zeros

train_data['text'] = train_data['text'].apply(encode_and_pad)
test_data['text']  = test_data['text'].apply(encode_and_pad)
```

**Before padding:**
```
Row 1: [1, 2, 3]        (length 3)
Row 2: [4, 5, 6, 7, 8]  (length 5)
```
**After padding (max_length=5):**
```
Row 1: [1, 2, 3, 0, 0]  (now length 5)
Row 2: [4, 5, 6, 7, 8]  (unchanged)
```

---

## Step 4 — Custom Dataset Class

Unlike torchvision (which handled this for images), with text you write your own `Dataset` class.

```python
from torch.utils.data import Dataset, DataLoader

class SentimentDataset(Dataset):
    def __init__(self, data):
        self.texts  = data['text'].values
        self.labels = data['label'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = self.texts[idx]
        label = self.labels[idx]
        return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.long)

train_dataset = SentimentDataset(train_data)
test_dataset  = SentimentDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)
```

**The 3 methods you must implement:**

| Method | What it does |
|---|---|
| `__init__` | Store the data |
| `__len__` | Return total number of samples |
| `__getitem__` | Return one sample as a tensor |

---

## Step 5 — Define the RNN Model

```python
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)   # word index → dense vector
        self.rnn       = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc        = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)                                   # [batch, seq_len] → [batch, seq_len, embed_size]
        h0 = torch.zeros(1, x.size(0), hidden_size)            # initial hidden state = all zeros
        out, _ = self.rnn(x, h0)                               # out: [batch, seq_len, hidden_size]
        out = self.fc(out[:, -1, :])                            # take ONLY the last timestep
        return out

# Hyperparameters
vocab_size  = len(vocab) + 1   # +1 for the padding index (0)
embed_size  = 128
hidden_size = 128
output_size = 2                 # positive / negative

model = SentimentRNN(vocab_size, embed_size, hidden_size, output_size)
```

**Key concepts:**

- `nn.Embedding(vocab_size, embed_size)` — converts integer word indices into dense vectors. Each word gets a learned vector of size `embed_size`. Think of it as a lookup table the model trains itself.
- `nn.RNN(embed_size, hidden_size, batch_first=True)` — the RNN layer. `batch_first=True` means input shape is `[batch, seq_len, features]` instead of the default `[seq_len, batch, features]`.
- `h0 = torch.zeros(1, batch_size, hidden_size)` — the initial hidden state (the RNN's "memory" starts empty).
- `out[:, -1, :]` — we only care about the **last** timestep's output, which has "seen" the whole sequence.

---

## Step 6 — Training Loop

Same 3-step ritual as always, just with text batches.

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for texts, labels in train_loader:
        outputs = model(texts)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')
```

---

## Step 7 — Evaluation

```python
model.eval()
correct = 0
total   = 0

with torch.no_grad():
    for texts, labels in test_loader:
        outputs = model(texts)
        _, predicted = torch.max(outputs.data, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')
```

---

## 🗺️ RNN vs ANN vs CNN: The Full Picture

| | ANN (Titanic) | CNN (MNIST) | RNN (IMDB) |
|---|---|---|---|
| Input type | Tabular (numbers) | 2D image | Sequential text |
| Data loading | pandas tensors | torchvision + DataLoader | Custom Dataset class |
| Preprocessing | StandardScaler | ToTensor() | Tokenize → Encode → Pad |
| Key new layer | `nn.Linear` | `nn.Conv2d` + Pool | `nn.Embedding` + `nn.RNN` |
| What the model "sees" | All features at once | Spatial patches | One token at a time |
| Output used | All neurons | Flattened feature map | **Last** timestep only |

---

## 📌 Quick Reference Card

| Thing | What it does |
|---|---|
| `str.lower().str.split()` | Tokenize: sentence → list of words |
| `vocab = set()` | Collect all unique words |
| `word_to_idx` | Dictionary: word string → integer index |
| `encode_and_pad(text)` | Convert word list to fixed-length integer list |
| Custom `Dataset` class | Wraps your data for use with DataLoader |
| `nn.Embedding(V, D)` | Lookup table: integer → dense vector of size D |
| `nn.RNN(input, hidden, batch_first=True)` | The RNN layer; processes sequence step by step |
| `h0 = torch.zeros(1, batch, hidden)` | Initial hidden state (the RNN's blank memory) |
| `out[:, -1, :]` | Final hidden state — the summary of the whole sequence |
| `LabelEncoder()` | Converts string labels ("positive"/"negative") to integers |