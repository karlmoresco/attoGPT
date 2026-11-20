import numpy as np
from attogpt.utils import softmax


def top_k_logits(logits, k):
    if k == 0 or k >= logits.shape[-1]:
        return logits

    kth_logit = np.sort(logits)[-k]
    return np.where(logits < kth_logit, -np.inf, logits)


def top_p_logits(logits, p, temperature=1.0):
    """Nucleus sampling"""
    if p >= 1.0:
        return logits
    
    ord = np.argsort(logits)[::-1]
    probs = softmax(logits, temperature=temperature)[ord]
    kept = np.cumsum(probs) - probs < p
    m = np.full_like(logits, -np.inf)
    m[ord[kept]] = logits[ord[kept]]
    return m


def sample(model, tokenizer, prompt, n_toks=75, temperature=1.0, 
           top_k=0, top_p=1.0, rng=None, stream=False):
    if rng is None:
        rng = np.random.default_rng()
    
    T = model.config.seq_len
    ids = tokenizer.encode(prompt)

    if stream:
        print(tokenizer.decode(ids), end="", flush=True)
    
    for _ in range(n_toks):
        if len(ids) >= T:
            window = np.asarray(ids[-T:], dtype=np.int64)[None, :]
            t = T - 1
        else:
            window = np.zeros((1, T), dtype=np.int64)
            window[0, :len(ids)] = ids
            t = len(ids) - 1
        
        logits = model.forward(window)[0, t]
        logits = top_k_logits(logits, top_k)
        logits = top_p_logits(logits, top_p, temperature=temperature)
        probs = softmax(logits, temperature=temperature)
        next_id = int(rng.choice(len(probs), p=probs))

        ids.append(next_id)
        if stream:
            print(tokenizer.decode([next_id]), end="", flush=True)
    
    return tokenizer.decode(ids)
