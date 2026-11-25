import subprocess
import sys

cmd = (
    'cd /kaggle/working && rm -rf INFERIO && '
    'git clone -q https://github.com/w-ahmad1a10/INFERIO.git && cd INFERIO && '
    'make -C engine -j 2>&1 | tail -2 && '
    'echo ==== WMMA PARITY ==== && '
    'CUDA_VISIBLE_DEVICES=1 INFERIO_ATTN=wmma python3 bench/parity_check.py '
    '--lib engine/build/libinferio.so --weights /kaggle/working/weights_gpt2 '
    '--golden /kaggle/working/golden --out temp/results/parity_wmma.json; '
    'echo "==== SCREEN: GPU0 simple vs GPU1 wmma ====" && '
    '( CUDA_VISIBLE_DEVICES=1 INFERIO_ATTN=wmma INFERIO_PROFILE=1 python3 tools/quick_compare.py '
    '--lib engine/build/libinferio.so --weights /kaggle/working/weights_gpt2 '
    '--batches 64,256,1024 --iters 2 > /kaggle/working/screen_wmma.txt 2>&1 ) & W=$!; '
    'CUDA_VISIBLE_DEVICES=0 INFERIO_ATTN=simple INFERIO_PROFILE=1 python3 tools/quick_compare.py '
    '--lib engine/build/libinferio.so --weights /kaggle/working/weights_gpt2 '
    '--batches 64,256,1024 --iters 2 > /kaggle/working/screen_simple.txt 2>&1; '
    'wait $W; '
    'echo ---- simple ----; grep -E "^B=|QUICK_JSON" /kaggle/working/screen_simple.txt | head -4; '
    'echo ---- wmma ----; grep -E "^B=|QUICK_JSON|failed" /kaggle/working/screen_wmma.txt | head -4'
)
r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=1700)
print(r.stdout[-9000:])
if r.stderr:
    print(r.stderr[-3000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)
