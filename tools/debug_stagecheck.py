#!/usr/bin/env python3
"""Stage-by-stage comparison of engine dumps vs the fp32 numpy reference.

Usage:
  INFERIO_DUMP_DIR=/kaggle/working/dump python3 bench/bench_prefill.py --single ...
  python3 tools/debug_stagecheck.py model.safetensors /kaggle/working/dump golden/ids_b2.npy
"""
import json
import os
import struct
import sys

import numpy as np

SEQ = 1024
EPS = 1e-5
H = 12
DH = 64
D = 768


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


def ln(x, w, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + EPS) * w + b


def gelu_new(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


def main():
    src, dump, ids_path = sys.argv[1], sys.argv[2], sys.argv[3]
    t = load_safetensors(src)
    ids = np.load(ids_path)
    B, S = ids.shape
    rows = B * S
    fails = []

    def load(name, shape):
        return np.fromfile(os.path.join(dump, name + ".bin"), dtype=np.float16) \
                 .reshape(shape).astype(np.float32)

    def cmp(name, ref):
        got = load(name, ref.shape)
        d = np.abs(got - ref)
        m = float(d.max())
        print(f"{name:16s} shape={ref.shape} max_abs_diff={m:.6f} mean={float(d.mean()):.6f}")
        if m > 1.0:
            fails.append(name)
        return got

    x = t["wte.weight"][ids] + t["wpe.weight"][:S]
    cmp("01_embed_x", x.reshape(rows, D))

    p = "h.0."
    xn = ln(x.reshape(rows, D), t[p + "ln_1.weight"], t[p + "ln_1.bias"])
    cmp("02_ln1_xn", xn)

    qkv = xn @ t[p + "attn.c_attn.weight"] + t[p + "attn.c_attn.bias"]
    cmp("03_qkv", qkv)

    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(B, S, H, DH).transpose(0, 2, 1, 3)
    k = k.reshape(B, S, H, DH).transpose(0, 2, 1, 3)
    v = v.reshape(B, S, H, DH).transpose(0, 2, 1, 3)
    sc = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(DH)
    mask = np.triu(np.ones((S, S), dtype=bool), 1)
    sc = np.where(mask, -1e30, sc)
    sc -= sc.max(-1, keepdims=True)
    pr = np.exp(sc)
    pr /= pr.sum(-1, keepdims=True)
    att = (pr @ v).transpose(0, 2, 1, 3).reshape(rows, D)
    cmp("04_att", att)

    x = x.reshape(rows, D) + att @ t[p + "attn.c_proj.weight"] + t[p + "attn.c_proj.bias"]
    cmp("05_proj_x", x)

    xn = ln(x, t[p + "ln_2.weight"], t[p + "ln_2.bias"])
    cmp("06_ln2_xn", xn)

    ffn = gelu_new(xn @ t[p + "mlp.c_fc.weight"] + t[p + "mlp.c_fc.bias"])
    cmp("07_fc1_gelu", ffn)

    x = x + ffn @ t[p + "mlp.c_proj.weight"] + t[p + "mlp.c_proj.bias"]
    cmp("08_fc2_x", x)

    xf = ln(x, t["ln_f.weight"], t["ln_f.bias"])
    cmp("09_final_xn", xf)

    hl = xf.reshape(B, S, D)[:, -1, :]
    cmp("10_hlast", hl)

    logits16 = np.fromfile(os.path.join(dump, "11_logits32.bin"), dtype=np.float32) \
                  .reshape(B, -1)
    ref = hl @ t["wte.weight"].T
    d = np.abs(logits16 - ref)
    print(f"{'11_logits32':16s} shape={ref.shape} max_abs_diff={float(d.max()):.6f} "
          f"mean={float(d.mean()):.6f}")
    print("argmax ref   :", ref.argmax(-1).tolist())
    print("argmax engine:", logits16.argmax(-1).tolist())
    if float(d.max()) > 1.0:
        fails.append("11_logits32")
    print("FAILS:", fails if fails else "none (all stages within 1.0 abs)")


if __name__ == "__main__":
    main()
