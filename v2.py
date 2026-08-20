import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
torch.manual_seed(1337)
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32  # number of embedding dimensions
# -------


with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)

# create a mapping from characters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)


# Train and Test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # 90 % partition
train_data = data[:n]  # 90% training data , rest validation
val_data = data[n:]  # 10% validation data


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )  # lower triangular matrix

        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # (B, T, head_size) @ (B, head_size, T) → (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) → (B, T, head_size)
        return out


# class MultiHeadAttention(nn.Module):
#     """multiple heads of self-attention in parallel"""

#     def __init__(self, num_heads, head_size):
#         super().__init__()
#         self.heads = nn.ModuleList(
#             [Head(head_size) for _ in range(num_heads)]
#         )  # list of heads

#     def forward(self, x):
#         return torch.cat(
#             [h(x) for h in self.heads], dim=-1
#         )  # concatenate outputs of all heads


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Below code think it as 2d matrix of vocab_size * vocab_size (in our case - 65 * 65)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.sa_head = Head(n_embd)  # self attention head

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)

        x = tok_emb + pos_emb  # (B, T, C)
        x = self.sa_head(x)  # apply one head of self attention (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss  # (B, T, C)

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[
                :, -block_size:
            ]  # crop context to the last block_size tokens
            # get the predictions
            logits, loss = self(
                idx_cond
            )  # inside generate → forward(idx, targets=None)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sample index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B , T+1)

        return idx


model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
    # sample a batch of data
    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)  # training loop  → forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # figures out how the parameters contributed to the error
    optimizer.step()  # actually changes those parameters

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
