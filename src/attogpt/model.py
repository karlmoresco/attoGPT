import numpy as np
from attogpt.utils import he_init, relu, normal_init
from attogpt.config import Config


class Attention:
    def __init__(
        self,
        config: Config,
        rng: np.random.Generator,
    ):
        self.config = config
        self.d_model = self.config.d_model
        self.n_heads = self.config.n_heads
        self.d_k = self.config.d_k

        self.attention_std = self.config.attention_std
        self.Wq = normal_init(self.d_model, self.d_model, self.attention_std, rng)
        self.Wk = normal_init(self.d_model, self.d_model, self.attention_std, rng)
        self.Wv = normal_init(self.d_model, self.d_model, self.attention_std, rng)


class MLP:
    def __init__(
        self,
        config: Config,
        rng: np.random.Generator,
    ):
        self.config = config
        self.d_model = self.config.d_model
        self.d_ff = self.config.d_ff

        self.W1 = he_init(self.d_model, self.d_ff, rng)
        self.b1 = np.zeros(self.d_ff)
        self.W2 = he_init(self.d_ff, self.d_model, rng)
        self.b2 = np.zeros(self.d_model)
    
    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.H1 = relu(self.Z1)
        self.Z2 = self.H1 @ self.W2 + self.b2
        return self.Z2


class LayerNorm:
    def __init__(
        self,
        config: Config,
        eps: float = 1e-5,
    ):
        self.config = config
        self.d_model = self.config.d_model
        self.eps = eps

        self.g = np.ones(self.d_model)
        self.b = np.zeros(self.d_model)

    def forward(self, x):
        self.x = x

        self.mu = np.mean(x, axis=-1, keepdims=True)
        self.var = np.mean((x - self.mu) ** 2, axis=-1, keepdims=True)

        self.x_bar = (x - self.mu) / np.sqrt(self.var + self.eps)
        return self.g * self.x_bar + self.b


class Embedding:
    def __init__(
        self,
        config: Config,
        rng: np.random.Generator,
    ):
        self.config = config
        self.E = normal_init(self.config.vocab_size, self.config.d_model, self.config.embedding_std, rng)
        self.token_ids = None

    def forward(self, token_ids):
        self.token_ids = token_ids
        return self.E[token_ids]


class AttoGPT():
    def __init__(
            self,
            config: Config,
            rng: np.random.Generator,
    ):
        self.config = config
        self.positional_encoding = self._get_positional_encoding()

    def _get_positional_encoding(self):
        """Sinusoidal positional encoder"""
        seq_len = self.config.seq_len
        d_model = self.config.d_model

        pe = np.zeros((seq_len, d_model))
        positions = np.arange(seq_len).reshape(-1, 1)

        N = 10_000

        factors = np.exp(-np.log(N) * np.arange(0, d_model, 2) / d_model).reshape(1, -1)  # (1, d_model / 2)
        pe[:, 0::2] = np.sin(positions * factors)
        pe[:, 1::2] = np.cos(positions * factors)

        return pe

    def encode_position(self, sequence):
        return sequence + self.positional_encoding
