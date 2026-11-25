import subprocess
import sys

# v0.10 screen: parity, then same-session A/B v0.9 vs v0.10 (alternating, GPU0).
cmd = (
    'cd /kaggle/working && rm -rf INFERIO && '
    'git clone -q https://github.com/w-ahmad1a10/INFERIO.git && cd INFERIO && '
    'make -C engine 2>&1 | grep -E "error" | head -10; '
    'ls -la engine/build/libinferio.so; cp engine/build/libinferio.so /kaggle/working/libinferio_v10.so && '
    'git checkout -q 3cc672f && make -C engine 2>&1 | grep -E "error" | head -5; '
    'cp engine/build/libinferio.so /kaggle/working/libinferio_v09.so && '
    'git checkout -q main && '
    'echo "==== PARITY v0.10 (GPU1) ====" && '
    'CUDA_VISIBLE_DEVICES=1 python3 bench/parity_check.py '
    '--lib /kaggle/working/libinferio_v10.so --weights /kaggle/working/weights_gpt2 '
    '--golden /kaggle/working/golden --out temp/results/parity_v010.json; '
    'echo "==== A/B GPU0: v09 v10 v09 v10 (B=64,256,1024; 1 warmup + 3 iters) ====" && '
    'for v in v09 v10 v09 v10; do '
    'echo "---- $v ----"; '
    'CUDA_VISIBLE_DEVICES=0 INFERIO_PROFILE=1 python3 tools/quick_compare.py '
    '--lib /kaggle/working/libinferio_$v.so --weights /kaggle/working/weights_gpt2 '
    '--batches 64,256,1024 --iters 3 --warmup 1 2>&1 | grep -E "^B=|failed" | head -3; '
    'done'
)
r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=1750)
print(r.stdout[-9000:])
if r.stderr:
    print(r.stderr[-2000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)