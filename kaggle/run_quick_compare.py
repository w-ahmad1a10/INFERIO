import subprocess
import sys

cmd = (
    'cd /kaggle/working && rm -rf INFERIO && '
    'git clone -q https://github.com/w-ahmad1a10/INFERIO.git && cd INFERIO && '
    'make -C engine 2>&1 | grep -E "error" | head -20; '
    'ls -la engine/build/libinferio.so; '
    'echo "==== MMA PARITY ====" && '
    'CUDA_VISIBLE_DEVICES=1 INFERIO_ATTN=mma python3 bench/parity_check.py '
    '--lib engine/build/libinferio.so --weights /kaggle/working/weights_gpt2 '
    '--golden /kaggle/working/golden --out temp/results/parity_mma.json; '
    'echo "==== SCREEN: GPU0 simple vs GPU1 mma ====" && '
    '( CUDA_VISIBLE_DEVICES=1 INFERIO_ATTN=mma INFERIO_PROFILE=1 python3 tools/quick_compare.py '
    '--lib engine/build/libinferio.so --weights /kaggle/working/weights_gpt2 '
    '--batches 64,256,1024 --iters 2 > /kaggle/working/screen_mma.txt 2>&1 ) & W=$!; '
    'CUDA_VISIBLE_DEVICES=0 INFERIO_ATTN=simple INFERIO_PROFILE=1 python3 tools/quick_compare.py '
    '--lib engine/build/libinferio.so --weights /kaggle/working/weights_gpt2 '
    '--batches 64,256,1024 --iters 2 > /kaggle/working/screen_simple.txt 2>&1; '
    'wait $W; '
    'echo ---- simple ----; grep -E "^B=|failed" /kaggle/working/screen_simple.txt | head -4; '
    'echo ---- mma ----; grep -E "^B=|failed" /kaggle/working/screen_mma.txt | head -4'
)
r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=1700)
print(r.stdout[-9000:])
if r.stderr:
    print(r.stderr[-3000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)
