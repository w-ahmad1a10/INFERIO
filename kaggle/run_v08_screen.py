import subprocess
import sys

cmd = (
    'cd /kaggle/working && rm -rf INFERIO && '
    'git clone -q https://github.com/w-ahmad1a10/INFERIO.git && cd INFERIO && '
    'make -C engine 2>&1 | grep -E "error" | head -20; '
    'ls -la engine/build/libinferio.so; '
    'echo "==== ptxas register report ====" && '
    'nvcc -O3 -std=c++17 -Xcompiler -fPIC -arch=sm_75 -Xptxas -v -Iengine -shared '
    'engine/v0/inferio_v0.cu -lcublas -lcublasLt -o /tmp/probe.so 2>&1 | '
    'grep -E "attn_mma_kernel|Used .* registers|spill" | head -8; '
    'echo "==== MMA PARITY (v0.8) ====" && '
    'CUDA_VISIBLE_DEVICES=1 python3 bench/parity_check.py '
    '--lib engine/build/libinferio.so --weights /kaggle/working/weights_gpt2 '
    '--golden /kaggle/working/golden --out temp/results/parity_v08.json; '
    'echo "==== SCREEN GPU0 (v0.8 mma) ====" && '
    'CUDA_VISIBLE_DEVICES=0 INFERIO_PROFILE=1 python3 tools/quick_compare.py '
    '--lib engine/build/libinferio.so --weights /kaggle/working/weights_gpt2 '
    '--batches 64,256,1024 --iters 2'
)
r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=1700)
print(r.stdout[-9000:])
if r.stderr:
    print(r.stderr[-3000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)
