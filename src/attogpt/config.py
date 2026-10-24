class Config:
    def __init__(
        self,
        vocab_size: int = 1000,  # Target number of tokens in vocabulary post-BPE,
        seq_len: int = 128,  # Number of tokens in a sequence
        d_model: int = 128,  # Vector dimension for token embeddings
        d_ff: int = 512,  # Matrix dimension for feedforward layers
        n_layers: int = 2,  # Number of transformer blocks
        n_heads: int = 4,  # Number of attention heads
        embeddings_std: float = 0.02,  # Standard deviation for initialization of embeddings
        attention_std: float = 0.02,  # Standard deviation for initialization of attention mechanism weights
        pe_scale: float = 0.1,  # Amplitude scale factor for sinusoidal positional encoding
        seed: int = 42,  # Random seed
    ):
        if d_model % 2 != 0:
            raise ValueError(
                "d_model must be an even integer"
            )
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embeddings_std = embeddings_std
        self.attention_std = attention_std
        self.pe_scale = pe_scale
        self.seed = seed

        self.d_k = d_model // n_heads
