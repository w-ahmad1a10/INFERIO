#!/usr/bin/env python3
"""PyTorch reference baselines — variants (NOT the frozen engine test).

Variants, all on the frozen ids / batch sizes / timing window of bench_prefill:
  eager-nocache : HF GPT-2 eager fp16 with use_cache=False (no KV allocation)
  compile       : torch.compile(transformer, dynamic=True) — Inductor/Triton
Run: python3 tools/bench_torch_variants.py --variant eager-nocache|compile
Writes results JSON + appends a LEDGER.md section marked reference-only.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
from bench_prefill import (BATCHES, ITERS, SEQ, WARMUP, ids_for_batch,  # noqa: E402
                           nvidia_snapshot)

import torch  # noqa: E402


def sha_self():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["eager-nocache", "compile"])
    ap.add_argument("--results-root", default="temp")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    from transformers import GPT2LMHeadModel

    dev = torch.device("cuda")
    print("torch", torch.__version__, "| cuda", torch.version.cuda, "|",
          torch.cuda.get_device_name(0), flush=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2").half().to(dev).eval()
    torch.set_grad_enabled(False)
    wte_t = model.transformer.wte.weight.t().contiguous()  # [768, 50257]

    transformer = model.transformer
    if args.variant == "compile":
        transformer = torch.compile(transformer, dynamic=True)

    def one(ids):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        ids_t = torch.from_numpy(ids).to(dev)
        h = transformer(ids_t, use_cache=False).last_hidden_state
        logits = h[:, -1, :] @ wte_t
        en.record()
        en.synchronize()
        return st.elapsed_time(en), logits

    run_id = time.strftime("%Y%m%d-%H%M%SZ", time.gmtime()) + f"-pytorch-{args.variant}"
    rows = []
    for b in BATCHES:
        ids = ids_for_batch(b)
        ids_sha = hashlib.sha256(ids.tobytes()).hexdigest()
        try:
            for _ in range(WARMUP):
                one(ids)
            ts = []
            for _ in range(ITERS):
                ms, logits = one(ids)
                ts.append(ms)
            arg = logits.argmax(-1).cpu().numpy()
            rows.append({"batch": b, "status": "ok", "median_ms": statistics.median(ts),
                         "min_ms": min(ts), "max_ms": max(ts), "latencies_ms": ts,
                         "tokens_per_s": b * SEQ / (statistics.median(ts) / 1000.0),
                         "ids_sha256": ids_sha, "argmax_first8": arg[:8].tolist()})
            print(f"B={b} ok median={statistics.median(ts):.1f} ms", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows.append({"batch": b, "status": "oom", "ids_sha256": ids_sha})
            print(f"B={b} oom", flush=True)
        del ids
        torch.cuda.empty_cache()

    results = {
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": f"pytorch-{args.variant}-{torch.__version__} (HF GPT-2, fp16, "
                  f"use_cache=False{' + torch.compile(dynamic=True)' if args.variant == 'compile' else ''})",
        "harness_sha256": sha_self(),
        "device_env": nvidia_snapshot(),
        "note": args.note,
        "rows": rows,
    }
    os.makedirs(os.path.join(args.results_root, "results"), exist_ok=True)
    path = os.path.join(args.results_root, "results", run_id + ".json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        f"\n## Run `{run_id}` — PyTorch {args.variant} GPT-2 fp16 — GPU 1",
        f"- torch {torch.__version__}, harness sha256 {results['harness_sha256'][:16]}…",
        f"- **Note:** {args.note or '(not specified)'} — reference baseline only, not the engine.",
        f"- device env: {'; '.join(results['device_env'])}",
        "",
        "| B | status | median ms | min ms | max ms | tokens/s |",
        "|---|--------|-----------|--------|--------|----------|",
    ]
    for r in rows:
        if r["status"] == "ok":
            lines.append(f"| {r['batch']} | ok | {r['median_ms']:.2f} | {r['min_ms']:.2f} | "
                         f"{r['max_ms']:.2f} | {r['tokens_per_s']:.1f} |")
        else:
            lines.append(f"| {r['batch']} | {r['status']} | – | – | – | – |")
    with open(os.path.join(args.results_root, "LEDGER.md"), "a") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path, "and appended LEDGER.md", flush=True)


if __name__ == "__main__":
    main()