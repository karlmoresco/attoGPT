import numpy as np
from attogpt.utils import he_init, relu, normal_init, softmax
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
        self.Wo = normal_init(self.d_model, self.d_model, self.attention_std, rng)

    def _split(self, x):
        B, T = x.shape[0], x.shape[1]
        x = x.reshape(B, T , self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (B, n_heads, T, d_k)
    
    def _concat(self, x):
        B, H, T, d_k = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * d_k)  # (B, T, d_model)

    def forward(self, x):
        self.x = x
        B, T = x.shape[0], x.shape[1]
        
        self.Q = self._split(x @ self.Wq)  # (B, H, T, d_k)
        self.K = self._split(x @ self.Wk)
        self.V = self._split(x @ self.Wv)

        scores = self.Q @ self.K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)  # (B, H, T, T)

        mask = np.triu(np.full((T, T), -1e10), k=1)
        masked_scores = mask + scores

        self.attn_pat = softmax(masked_scores, axis=-1)
        head_out = self.attn_pat @ self.V

        self.concat = self._concat(head_out)  # (B, T, d_model)
        return self.concat @ self.Wo

    def backward(self, dout):
        B, T, _ = dout.shape
        N = B * T

        self.dWo = self.concat.reshape(N, -1).T @ dout.reshape(N, -1)
        dconcat = dout @ self.Wo.T

        dhead_out = self._split(dconcat)  # (B, H, T, d_k)
        dV = self.attn_pat.transpose(0, 1, 3, 2) @ dhead_out
        dattn_pat = dhead_out @ self.V.transpose(0, 1, 3, 2)

        D = np.sum(dattn_pat * self.attn_pat, axis=-1, keepdims=True)  # (B, H, T, 1)
        dscores = self.attn_pat * (dattn_pat - D)

        dQ = (dscores @ self.K) / np.sqrt(self.d_k)
        dK = (dscores.transpose(0, 1, 3, 2) @ self.Q) / np.sqrt(self.d_k)

        dxWq = self._concat(dQ)
        dxWk = self._concat(dK)
        dxWv = self._concat(dV)

        self.dWq = self.x.reshape(N, -1).T @ dxWq.reshape(N, -1)
        self.dWk = self.x.reshape(N, -1).T @ dxWk.reshape(N, -1)
        self.dWv = self.x.reshape(N, -1).T @ dxWv.reshape(N, -1)

        dx = dxWq @ self.Wq.T + dxWk @ self.Wk.T + dxWv @ self.Wv.T
        return dx


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
    
    def backward(self, dout):
        B, T, _ = dout.shape
        N = B * T

        self.dW2 = self.H1.reshape(N, -1).T @ dout.reshape(N, -1)
        self.db2 = np.sum(dout, axis=(0, 1))

        dH1 = dout @ self.W2.T
        dZ1 = dH1 * (self.Z1 > 0)

        self.dW1 = self.X.reshape(N, -1).T @ dZ1.reshape(N, -1)
        self.db1 = np.sum(dZ1, axis=(0, 1))

        dx = dZ1 @ self.W1.T
        return dx



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
    
    def backward(self, dout):
        self.dg = np.sum(dout * self.x_bar, axis=(0, 1))
        self.db = np.sum(dout, axis=(0, 1))

        s = np.sqrt(self.var + self.eps)  # (B, T, 1)
        c = dout * self.g  # (B, T, d_model)
        mean_c = np.mean(c, axis=-1, keepdims=True)
        mean_cx = np.mean(c * self.x_bar, axis=-1, keepdims=True)
        dx = (c - mean_c - self.x_bar * mean_cx) / s  # (B, T, d_model)
        return dx


class Embedding:
    def __init__(
        self,
        config: Config,
        rng: np.random.Generator,
    ):
        self.config = config
        self.W_emb = normal_init(self.config.vocab_size, self.config.d_model, self.config.embeddings_std, rng)
        self.token_ids = None

    def forward(self, token_ids):
        self.token_ids = token_ids  # (B, T)
        return self.W_emb[token_ids]
    
    def backward(self, dout):
        self.dW_emb = np.zeros_like(self.W_emb)
        np.add.at(self.dW_emb, self.token_ids, dout)


class Block:
    def __init__(
        self,
        config: Config,
        rng: np.random.Generator,
    ):
        self.ln_1 = LayerNorm(config)
        self.attn = Attention(config, rng)
        self.ln_2 = LayerNorm(config)
        self.mlp = MLP(config, rng)
    
    def forward(self, x):
        """Pre-LN transformer block"""
        x = x + self.attn.forward(self.ln_1.forward(x))
        x = x + self.mlp.forward(self.ln_2.forward(x))
        return x

    def backward(self, dout):
        dU = dout + self.ln_2.backward(self.mlp.backward(dout))
        dx = dU + self.ln_1.backward(self.attn.backward(dU))
        return dx


class AttoGPT:
    def __init__(
            self,
            config: Config,
            rng: np.random.Generator,
    ):
        self.config = config
        self.embedding = Embedding(config, rng)
        self.positional_encoding = self._get_positional_encoding()
        self.blocks = [Block(config, rng) for _ in range(self.config.n_layers)]
        self.ln_f = LayerNorm(config)
        self.W_out = normal_init(self.config.d_model, self.config.vocab_size, self.config.embeddings_std, rng)

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

        return self.config.pe_scale * pe

    def encode_position(self, x):
        return x + self.positional_encoding

    def forward(self, token_ids):
        x = self.embedding.forward(token_ids)
        x = self.encode_position(x)
        for block in self.blocks:
            x = block.forward(x)
        self.h = self.ln_f.forward(x)
        self.logits = self.h @ self.W_out
        return self.logits

    def backward(self, targets):
        B, T, V = self.logits.shape
        N = B * T

        probs = softmax(self.logits, axis=-1)

        one_hot = np.zeros_like(probs)
        b_idx = np.arange(B)[:, None]
        t_idx = np.arange(T)[None, :]
        one_hot[b_idx, t_idx, targets] = 1.0

        G = (probs - one_hot) / N
        H = self.h.reshape(N, -1)
        self.dW_out = H.T @ G.reshape(N, V)

        dh = G @ self.W_out.T  # (B, T, d_model)
        dx = self.ln_f.backward(dh)

        for block in reversed(self.blocks):
            dx = block.backward(dx)

        self.embedding.backward(dx)
