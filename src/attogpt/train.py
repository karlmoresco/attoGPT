import numpy as np
from attogpt.config import Config


class Train:
    def __init__(
            self,
            model,
            config: Config,
            corpus,
            rng: np.random.Generator,
            batch_size: int = 32,
            lr: float = 2e-4,
            steps: int = 4000,
    ):
        self.model = model
        self.config = config
        self.corpus = np.asarray(corpus)
        self.rng = rng
        self.seq_len = self.config.seq_len
        self.batch_size = batch_size
        self.lr = lr
        self.steps = steps
    
    def batch(self):
        startpoints = self.rng.integers(0, len(self.corpus) - self.seq_len, size=self.batch_size)
        x = np.stack([self.corpus[sp : sp + self.seq_len] for sp in startpoints])
        y = np.stack([self.corpus[sp + 1 : sp + self.seq_len + 1] for sp in startpoints])
        return x, y

    def step(self):
        token_ids, targets = self.batch()
        self.model.forward(token_ids)
        self.model.backward(targets)

        self.model.W_out -= self.lr * self.model.dW_out

        self.model.ln_f.g -= self.lr * self.model.ln_f.dg
        self.model.ln_f.b -= self.lr * self.model.ln_f.db

        for block in self.model.blocks:
            block.ln_1.g -= self.lr * block.ln_1.dg
            block.ln_1.b -= self.lr * block.ln_1.db
            block.attn.Wq -= self.lr * block.attn.dWq
            block.attn.Wk -= self.lr * block.attn.dWk
            block.attn.Wv -= self.lr * block.attn.dWv
            block.attn.Wo -= self.lr * block.attn.dWo
            block.ln_2.g -= self.lr * block.ln_2.dg
            block.ln_2.b -= self.lr * block.ln_2.db
            block.mlp.W1 -= self.lr * block.mlp.dW1
            block.mlp.b1 -= self.lr * block.mlp.db1
            block.mlp.W2 -= self.lr * block.mlp.dW2
            block.mlp.b2 -= self.lr * block.mlp.db2
        
        self.model.embedding.W_emb -= self.lr * self.model.embedding.dW_emb
