import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================
# MODELLO GPT MINIMAL
# ============================================================

class GPTConfig:
    def __init__(self, vocab_size, d_model=128, context_length=64, n_layers=2, n_heads=4):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.context_length = context_length
        self.n_layers = n_layers
        self.n_heads = n_heads

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.ReLU(),
            nn.Linear(config.d_model * 4, config.d_model)
        )
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("mask", mask == 0)

    def forward(self, x):
        B, T, C = x.shape
        attn_mask = self.mask[:T, :T]
        out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.ln1(x + out)
        x = self.ln2(x + self.ffn(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.config = config

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=40, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.context_length:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

# ============================================================
# VOCABOLARIO
# ============================================================

def build_vocab(text):
    words = list(set(text.split()))
    vocab = {w: i for i, w in enumerate(words)}
    vocab["<UNK>"] = len(vocab)
    return vocab

def encode(text, vocab):
    return [vocab.get(w, vocab["<UNK>"]) for w in text.split()]

def decode(tokens, inv_vocab):
    return " ".join(inv_vocab[t] for t in tokens if t in inv_vocab)

# ============================================================
# TRAINING
# ============================================================

def get_batch(data, batch_size, ctx, device):
    ix = torch.randint(0, len(data) - ctx - 1, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+ctx]) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+1+ctx]) for i in ix])
    return x.to(device), y.to(device)

def train_model(dataset_path="data/dataset_dialogo.txt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw = open(dataset_path, "r", encoding="utf-8").read()
    vocab = build_vocab(raw)
    inv_vocab = {i: w for w, i in vocab.items()}
    tokens = encode(raw, vocab)

    config = GPTConfig(vocab_size=len(vocab))
    model = GPT(config).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    epochs = 5
    steps = 300
    batch_size = 32

    for e in range(epochs):
        total = 0
        for s in range(steps):
            x, y = get_batch(tokens, batch_size, config.context_length, device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.view(-1, config.vocab_size), y.view(-1))
            loss.backward()
            opt.step()
            total += loss.item()

        avg = total / steps
        losses.append(avg)
        print(f"Epoch {e+1}/{epochs} Loss: {avg:.4f}")

    torch.save({"model": model.state_dict(), "vocab": vocab, "config": config.__dict__}, "gpt_model.pth")
    print("Modello salvato in gpt_model.pth")

    plt.plot(losses)
    plt.show()

    return model, vocab, inv_vocab, config

# ============================================================
# CHAT
# ============================================================

def chat(model, vocab, inv_vocab, config):
    print("\nChat attiva! Scrivi qualcosa.\n")

    while True:
        user = input("Tu: ").strip()
        prompt = f"USER: {user}\nASSISTANT:"
        ids = encode(prompt, vocab)
        x = torch.tensor([ids], dtype=torch.long)

        out = model.generate(x, 40)
        out = out[0].tolist()
        reply = decode(out[len(ids):], inv_vocab)

        # 🔥 BLOCCO CHE RISOLVE IL TUO PROBLEMA
        if "USER:" in reply:
            reply = reply.split("USER:")[0].strip()

        print("IA:", reply)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    model, vocab, inv_vocab, config = train_model("data/dataset_dialogo.txt")
    chat(model, vocab, inv_vocab, config)
