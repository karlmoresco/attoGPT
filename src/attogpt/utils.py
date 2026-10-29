import regex as re
import numpy as np


def relu(x):
    return np.maximum(0, x)


def normal_init(m, n, std, rng):
    """Normal distribution initialization (0-centered)"""
    return rng.standard_normal(size=(m, n)) * std


def he_init(fan_in, fan_out, rng):
    """He initialization"""
    return rng.standard_normal(size=(fan_in, fan_out)) * np.sqrt(2 / fan_in)


def softmax(x, axis=-1, temperature=1.0):
    """Softmax, shifted for numerical stability."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = x / temperature
    shifted = scaled - np.max(scaled, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def cross_entropy(logits, targets, eps=1e-12):
    B, T, _ = logits.shape

    probs = softmax(logits, axis=-1)

    b_idx = np.arange(B)[:, None]
    t_idx = np.arange(T)[None, :]
    target_probs = probs[b_idx, t_idx, targets]  # (B, T)

    loss = -np.mean(np.log(target_probs + eps))
    return loss


def clean_dataset(input_path, output_path):
    """Clean raw data corpus"""
    with open(input_path, "r", encoding="utf-8") as f_in:
        data = f_in.read()

    data = re.sub(r'\s+', ' ', data)
    data = data.strip()

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write(data)

    return data
