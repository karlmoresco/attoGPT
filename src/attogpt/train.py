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

