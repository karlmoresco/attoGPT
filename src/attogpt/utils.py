import json
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


def softmax(x, axis=-1):
    """Softmax, shifted for numerical stability."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def clean_dataset(input_path, output_path):
    """Clean raw data corpus"""
    with open(input_path, "r", encoding="utf-8") as f_in:
        data = f_in.read()

    data = re.sub(r'\s+', ' ', data)
    data = data.strip()

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write(data)

    return data
