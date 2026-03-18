from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.0
