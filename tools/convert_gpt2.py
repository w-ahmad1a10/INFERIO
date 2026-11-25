#!/usr/bin/env python3
"""Download GPT-2 (124M) and convert it to INFERIO's flat fp16 tensor layout.

No PyTorch: the safetensors container is parsed directly with numpy.
Output: <out_dir>/<tensor>.f16 (row-major fp16) + manifest.txt + meta.json

Manifest format, one line per tensor:  <name> <rows> <cols> <bytes>
1-D tensors (biases / layernorm gains) are written with rows=1.
"""
import hashlib
import json
import os
import struct
import sys
import urllib.request

import numpy as np

MODEL_URL = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        start = 8 + n
        f.seek(start)
        raw = f.read()
    tensors = {}
    for name, info in hdr.items():
        if name == "__metadata__":
            continue
        b, e = info["data_offsets"]
        if info["dtype"] != "F32":
            raise ValueError(f"unexpected dtype {info['dtype']} for {name}")
        tensors[name] = np.frombuffer(raw[b:e], dtype=np.float32).reshape(info["shape"])
    return tensors


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/weights_gpt2"
    src = sys.argv[2] if len(sys.argv) > 2 else "/kaggle/working/model.safetensors"
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(src) or os.path.getsize(src) < 400_000_000:
        print(f"downloading {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, src)

    h = hashlib.sha256()
    with open(src, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    print("safetensors sha256:", h.hexdigest(), "bytes:", os.path.getsize(src))

    t = load_safetensors(src)

    out = {
        "wte": t["wte.weight"],
        "wpe": t["wpe.weight"],
        "lnf_w": t["ln_f.weight"],
        "lnf_b": t["ln_f.bias"],
    }
    for i in range(12):
        p = f"h.{i}."
        out[f"h{i}.ln1_w"] = t[p + "ln_1.weight"]
        out[f"h{i}.ln1_b"] = t[p + "ln_1.bias"]
        out[f"h{i}.qkv_w"] = t[p + "attn.c_attn.weight"]
        out[f"h{i}.qkv_b"] = t[p + "attn.c_attn.bias"]
        out[f"h{i}.proj_w"] = t[p + "attn.c_proj.weight"]
        out[f"h{i}.proj_b"] = t[p + "attn.c_proj.bias"]
        out[f"h{i}.ln2_w"] = t[p + "ln_2.weight"]
        out[f"h{i}.ln2_b"] = t[p + "ln_2.bias"]
        out[f"h{i}.fc1_w"] = t[p + "mlp.c_fc.weight"]
        out[f"h{i}.fc1_b"] = t[p + "mlp.c_fc.bias"]
        out[f"h{i}.fc2_w"] = t[p + "mlp.c_proj.weight"]
        out[f"h{i}.fc2_b"] = t[p + "mlp.c_proj.bias"]

    manifest = []
    total = 0
    for name, arr in out.items():
        a = np.ascontiguousarray(arr.astype(np.float16))
        if a.ndim == 2:
            rows, cols = a.shape
        else:
            rows, cols = 1, a.shape[0]
        a.tofile(os.path.join(out_dir, name + ".f16"))
        manifest.append(f"{name} {rows} {cols} {a.nbytes}")
        total += a.nbytes
    with open(os.path.join(out_dir, "manifest.txt"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    meta = {
        "source": MODEL_URL,
        "safetensors_sha256": h.hexdigest(),
        "total_fp16_bytes": total,
        "n_tensors": len(out),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {len(out)} tensors, {total / 1e6:.1f} MB fp16 -> {out_dir}")


if __name__ == "__main__":
    main()
