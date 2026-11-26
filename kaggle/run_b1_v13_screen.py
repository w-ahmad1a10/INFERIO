import subprocess
import sys

# v0.13.0 screen (B=1 campaign): build, parity (direct path), A/B graph vs
# v0.12.2 at B=1, plus one b1-bench run to verify the graph path's argmax.
cmd = (
    'cd /kaggle/working/INFERIO && cp engine/build/libinferio.so /kaggle/working/libinferio_v12.so && '
    'cd /kaggle/working && rm -rf INFERIO && '
    'git clone -q https://github.com/w-ahmad1a10/INFERIO.git && cd INFERIO && '
    'make -C engine 2>&1 | grep -E "error" | head -15; '
    'ls -la engine/build/libinferio.so; cp engine/build/libinferio.so /kaggle/working/libinferio_v13.so && '
    'echo "==== PARITY v0.13 B=1 (GPU1, direct path) ====" && '
    'CUDA_VISIBLE_DEVICES=1 python3 bench/parity_check.py --batch 1 '
    '--lib /kaggle/working/libinferio_v13.so --weights /kaggle/working/weights_gpt2 '
    '--golden /kaggle/working/golden --out temp/results/parity_b1_v013.json; '
    'echo "==== b1 bench v0.13 (graph path argmax check) ====" && '
    'CUDA_VISIBLE_DEVICES=1 INFERIO_COMMIT=pre-graph-check python3 b1/bench_b1.py '
    '--lib /kaggle/working/libinferio_v13.so --weights /kaggle/working/weights_gpt2 '
    '--note "graph-path argmax verification (pre-screen)" | tail -3; '
    'echo "==== A/B GPU0: v12 v13 v12 v13 (B=1; 1 warmup + 3 iters) ====" && '
    'for v in v12 v13 v12 v13; do '
    'echo "---- $v ----"; '
    'CUDA_VISIBLE_DEVICES=0 python3 tools/quick_compare.py '
    '--lib /kaggle/working/libinferio_$v.so --weights /kaggle/working/weights_gpt2 '
    '--batches 1 --iters 3 --warmup 1 2>&1 | grep -E "^B=|failed" | head -2; '
    'done'
)
r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=1700)
print(r.stdout[-9000:])
if r.stderr:
    print(r.stderr[-2000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)