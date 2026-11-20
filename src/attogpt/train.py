import numpy as np
import pickle
import os
from attogpt.config import Config
from attogpt.utils import cross_entropy
from attogpt.opt import Adam


class Train:
    def __init__(
            self,
            model,
            config: Config,
            corpus,
            rng: np.random.Generator,
            batch_size: int = 32,
            lr: float = 2e-4,
            steps: int = 5000,
            split_factor: float = 0.90,
            val_seed: int | None = None,
            val_batches: int = 20,
    ):
        self.model = model
        self.config = config
        self.corpus = np.asarray(corpus)
        self.rng = rng
        self.seq_len = self.config.seq_len
        self.batch_size = batch_size
        self.lr = lr
        self.steps = steps

        self.train_losses = []
        self.val_losses = []

        cut = int(split_factor * len(self.corpus))
        self.train_ids = self.corpus[:cut]
        self.val_ids = self.corpus[cut:]

        self.val_seed = val_seed if val_seed is not None else self.config.seed
        self.val_batches = val_batches

        self.optimizer = Adam(self._params(), lr=self.lr)
    
    @classmethod
    def from_checkpoint(cls, path, corpus, **overrides):
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
    
        model = ckpt["model"]
        rng = np.random.default_rng()
        rng.bit_generator.state = ckpt["rng_state"]

        trainer = cls(model, model.config, corpus, rng, **overrides)
        trainer.train_losses = ckpt["train_losses"]
        trainer.val_losses = ckpt["val_losses"]

        trainer.optimizer.m = ckpt["optimizer"].m
        trainer.optimizer.v = ckpt["optimizer"].v
        trainer.optimizer.t = ckpt["optimizer"].t

        return trainer
    
    def _params(self):
        mdl = self.model
        params = [mdl.W_out, mdl.ln_f.g, mdl.ln_f.b]
        for block in mdl.blocks:
            params += [
                block.ln_1.g, block.ln_1.b,
                block.attn.Wq, block.attn.Wk, block.attn.Wv, block.attn.Wo,
                block.ln_2.g, block.ln_2.b,
                block.mlp.W1, block.mlp.b1, block.mlp.W2, block.mlp.b2 
            ]
        params += [mdl.embedding.W_emb]
        return params

    def _grads(self):
        mdl = self.model
        grads = [mdl.dW_out, mdl.ln_f.dg, mdl.ln_f.db]
        for block in mdl.blocks:
            grads += [
                block.ln_1.dg, block.ln_1.db,
                block.attn.dWq, block.attn.dWk, block.attn.dWv, block.attn.dWo,
                block.ln_2.dg, block.ln_2.db,
                block.mlp.dW1, block.mlp.db1, block.mlp.dW2, block.mlp.db2 
            ]
        grads += [mdl.embedding.dW_emb]
        return grads

    def batch(self, corpus, rng):
        startpoints = rng.integers(0, len(corpus) - self.seq_len, size=self.batch_size)
        x = np.stack([corpus[sp : sp + self.seq_len] for sp in startpoints])
        y = np.stack([corpus[sp + 1 : sp + self.seq_len + 1] for sp in startpoints])
        return x, y

    def step(self):
        token_ids, targets = self.batch(self.train_ids, self.rng)
        logits = self.model.forward(token_ids)
        loss = cross_entropy(logits, targets)
        self.model.backward(targets)

        self.optimizer.step(self._grads())

        return loss

    def train(self, log_factor=2, eval_freq=250, patience=15, save_path=None):
        best_val = float("inf")
        stale = 0
        log_freq = log_factor * eval_freq

        for i in range(self.steps):
            loss = self.step()
            self.train_losses.append(loss)

            if i % eval_freq == 0:
                val_loss = self.evaluate()
                self.val_losses.append((i, val_loss))

                if val_loss < best_val:
                    best_val = val_loss
                    stale = 0
                    if save_path is not None:
                        self.save(save_path)
                else:
                    stale += 1
                    if stale >= patience:
                        print(f"stopped at step {i} - best val loss: {best_val:.5f}")
                        break

                if i % log_freq == 0:
                    avg = np.mean(self.train_losses[-log_freq:])
                    print(f"step {i}/{self.steps} - loss: {loss:.5f} (avg {avg:.5f}), "
                          f"val: {val_loss:.5f} (best {best_val:.5f})")
    

    def evaluate(self):
        val_rng = np.random.default_rng(self.val_seed)
        losses = []
        for _ in range(self.val_batches):
            token_ids, targets = self.batch(self.val_ids, val_rng)
            logits = self.model.forward(token_ids)
            losses.append(cross_entropy(logits, targets))
        return float(np.mean(losses))
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "model": self.model,
            "optimizer": self.optimizer,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "rng_state": self.rng.bit_generator.state,
        }
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)
