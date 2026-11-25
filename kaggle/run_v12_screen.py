import subprocess
import sys

# v0.12 screen: ptxas, parity (default + mma128), A/B mma vs mma128 at B=1024,
# and v09 vs v12-default at B=256 (group tweak effect).
cmd = (
    'cd /kaggle/working && rm -rf INFERIO && '
    'git clone -q https://github.com/w-ahmad1a10/INFERIO.git && cd INFERIO && '
    'echo "==== ptxas ====" && '
    'nvcc -O3 -std=c++17 -Xcompiler -fPIC -arch=sm_75 -Xptxas -v -Iengine -shared '
    'engine/v0/inferio_v0.cu -lcublas -lcublasLt -o /tmp/probe.so 2>&1 | '
    'grep -E "attn_mma|registers|spill" | head -8; '
    'make -C engine 2>&1 | grep -E "error" | head -10; '
    'cp engine/build/libinferio.so /kaggle/working/libinferio_v12.so && '
    'echo "==== PARITY v0.12 mma (GPU1) ====" && '
    'CUDA_VISIBLE_DEVICES=1 python3 bench/parity_check.py '
    '--lib /kaggle/working/libinferio_v12.so --weights /kaggle/working/weights_gpt2 '
    '--golden /kaggle/working/golden --out temp/results/parity_v012.json; '
    'echo "==== PARITY v0.12 mma128 (GPU1) ====" && '
    'CUDA_VISIBLE_DEVICES=1 INFERIO_ATTN=mma128 python3 bench/parity_check.py '
    '--lib /kaggle/working/libinferio_v12.so --weights /kaggle/working/weights_gpt2 '
    '--golden /kaggle/working/golden --out temp/results/parity_v012_mma128.json; '
    'echo "==== A/B GPU0: mma vs mma128 @1024 | v09 vs v12 @256 ====" && '
    'for v in mma mma128 mma mma128; do '
    'echo "---- attn=$v B=1024 ----"; '
    'CUDA_VISIBLE_DEVICES=0 INFERIO_ATTN=$v INFERIO_PROFILE=1 python3 tools/quick_compare.py '
    '--lib /kaggle/working/libinferio_v12.so --weights /kaggle/working/weights_gpt2 '
    '--batches 1024 --iters 3 --warmup 1 2>&1 | grep -E "^B=|failed" | head -2; '
    'done; '
    'for v in v09 v12 v09 v12; do '
    'echo "---- $v B=256 (group tweak) ----"; '
    'CUDA_VISIBLE_DEVICES=0 INFERIO_PROFILE=1 python3 tools/quick_compare.py '
    '--lib /kaggle/working/libinferio_$v.so --weights /kaggle/working/weights_gpt2 '
    '--batches 256 --iters 3 --warmup 1 2>&1 | grep -E "^B=|failed" | head -2; '
    'done'
)
r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=1750)
print(r.stdout[-9000:])
if r.stderr:
    print(r.stderr[-2000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)