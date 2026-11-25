import subprocess
import sys

# Same-session A/B: build v0.8 (21ca098) and v0.9 (HEAD) .so side by side,
# alternate on GPU0 with 3 iters each. Controls Kaggle session drift.
cmd = (
    'cd /kaggle/working && rm -rf INFERIO && '
    'git clone -q https://github.com/w-ahmad1a10/INFERIO.git && cd INFERIO && '
    'make -C engine 2>&1 | grep -E "error" | head -5; cp engine/build/libinferio.so /kaggle/working/libinferio_v09.so && '
    'git checkout -q 21ca098 && make -C engine 2>&1 | grep -E "error" | head -5; '
    'cp engine/build/libinferio.so /kaggle/working/libinferio_v08.so && '
    'git checkout -q main && '
    'echo "==== A/B GPU0: v08 v09 v08 v09 (B=64,256,1024; 1 warmup + 3 iters) ====" && '
    'for v in v08 v09 v08 v09; do '
    'echo "---- $v ----"; '
    'CUDA_VISIBLE_DEVICES=0 INFERIO_PROFILE=1 python3 tools/quick_compare.py '
    '--lib /kaggle/working/libinferio_$v.so --weights /kaggle/working/weights_gpt2 '
    '--batches 64,256,1024 --iters 3 --warmup 1 2>&1 | grep -E "^B=|failed" | head -3; '
    'done'
)
r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=1750)
print(r.stdout[-8000:])
if r.stderr:
    print(r.stderr[-2000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)