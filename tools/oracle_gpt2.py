#!/usr/bin/env python3
"""Float32 numpy reference forward pass for GPT-2 (no PyTorch).

Generates the frozen benchmark inputs for a batch size B and the golden
last-position logits used as the correctness oracle for the CUDA engine.

Frozen input rule (must match bench/SPEC.md exactly):
    rng = numpy.random.default_rng(20261125 + B)
    ids = rng.integers(0, 50257, size=(B, 1024), dtype=int32)
"""
import hashlib
import json
import struct
import sys

import numpy as np

SEED_BASE = 20261125
VOCAB = 50257
SEQ = 1024
N_LAYER = 12
N_HEAD = 12
D = 64
EPS = 1e-5


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        f.seek(8 + n)
        raw = f.read()
    t = {}
    for name, info in hdr.items():
        if name == "__metadata__":
            continue
        b, e = info["data_offsets"]
        t[name] = np.frombuffer(raw[b:e], dtype=np.float32).reshape(info["shape"])
    return t


def ids_for_batch(batch):
    rng = np.random.default_rng(SEED_BASE + batch)
    return rng.integers(0, VOCAB, size=(batch, SEQ), dtype=np.int32)


def layer_norm(x, w, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + EPS) * w + b


def gelu_new(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


def forward(t, ids):
    batch = ids.shape[0]
    x = t["wte.weight"][ids] + t["wpe.weight"][:SEQ]
    for i in range(N_LAYER):
        p = f"h.{i}."
        h = layer_norm(x, t[p + "ln_1.weight"], t[p + "ln_1.bias"])
        qkv = h @ t[p + "attn.c_attn.weight"] + t[p + "attn.c_attn.bias"]
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(batch, SEQ, N_HEAD, D).transpose(0, 2, 1, 3)
        k = k.reshape(batch, SEQ, N_HEAD, D).transpose(0, 2, 1, 3)
        v = v.reshape(batch, SEQ, N_HEAD, D).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(D)
        mask = np.triu(np.ones((SEQ, SEQ), dtype=bool), 1)
        scores = np.where(mask, -1e30, scores)
        scores -= scores.max(-1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(-1, keepdims=True)
        att = (probs @ v).transpose(0, 2, 1, 3).reshape(batch, SEQ, N_HEAD * D)
        x = x + att @ t[p + "attn.c_proj.weight"] + t[p + "attn.c_proj.bias"]
        h = layer_norm(x, t[p + "ln_2.weight"], t[p + "ln_2.bias"])
        f = gelu_new(h @ t[p + "mlp.c_fc.weight"] + t[p + "mlp.c_fc.bias"])
        x = x + f @ t[p + "mlp.c_proj.weight"] + t[p + "mlp.c_proj.bias"]
    x = layer_norm(x, t["ln_f.weight"], t["ln_f.bias"])
    logits = x[:, -1, :] @ t["wte.weight"].T
    return logits.astype(np.float32)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/model.safetensors"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/kaggle/working/golden"
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    t = load_safetensors(src)
    ids = ids_for_batch(batch)
    logits = forward(t, ids)

    import os

    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/ids_b{batch}.npy", ids)
    np.save(f"{out_dir}/last_logits_b{batch}.npy", logits)
    meta = {
        "batch": batch,
        "seq": SEQ,
        "seed_rule": "default_rng(20261125 + B)",
        "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
        "logits_shape": list(logits.shape),
    }
    with open(f"{out_dir}/meta_b{batch}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print("argmax:", logits.argmax(-1).tolist())


if __name__ == "__main__":
    main()
