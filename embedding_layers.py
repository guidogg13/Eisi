import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, dim, max_seq_len):
        super().__init__()

        # Embedding dei token
        self.token_embedding = nn.Embedding(vocab_size, dim)

        # Embedding delle posizioni
        self.position_embedding = nn.Embedding(max_seq_len, dim)

        # Dropout (opzionale ma utile)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        x: tensor di shape (batch, seq_len)
        """

        batch_size, seq_len = x.shape

        # Crea i numeri delle posizioni: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # Ottieni gli embedding
        tok_emb = self.token_embedding(x)          # (batch, seq_len, dim)
        pos_emb = self.position_embedding(positions)  # (1, seq_len, dim)

        # Somma token + posizione
        out = tok_emb + pos_emb

        return self.dropout(out)
