import subprocess
import sys

# v0.11 screen: ptxas register/spill report, parity, same-session A/B vs v0.9.
cmd = (
    'cd /kaggle/working && rm -rf INFERIO && '
    'git clone -q https://github.com/w-ahmad1a10/INFERIO.git && cd INFERIO && '
    'echo "==== ptxas ====" && '
    'nvcc -O3 -std=c++17 -Xcompiler -fPIC -arch=sm_75 -Xptxas -v -Iengine -shared '
    'engine/v0/inferio_v0.cu -lcublas -lcublasLt -o /tmp/probe.so 2>&1 | '
    'grep -E "attn_mma_kernel|registers|spill" | head -6; '
    'make -C engine 2>&1 | grep -E "error" | head -10; '
    'cp engine/build/libinferio.so /kaggle/working/libinferio_v11.so && '
    'echo "==== PARITY v0.11 (GPU1) ====" && '
    'CUDA_VISIBLE_DEVICES=1 python3 bench/parity_check.py '
    '--lib /kaggle/working/libinferio_v11.so --weights /kaggle/working/weights_gpt2 '
    '--golden /kaggle/working/golden --out temp/results/parity_v011.json; '
    'echo "==== A/B GPU0: v09 v11 v09 v11 (B=256,1024; 1 warmup + 3 iters) ====" && '
    'for v in v09 v11 v09 v11; do '
    'echo "---- $v ----"; '
    'CUDA_VISIBLE_DEVICES=0 INFERIO_PROFILE=1 python3 tools/quick_compare.py '
    '--lib /kaggle/working/libinferio_$v.so --weights /kaggle/working/weights_gpt2 '
    '--batches 256,1024 --iters 3 --warmup 1 2>&1 | grep -E "^B=|failed" | head -3; '
    'done'
)
r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=1750)
print(r.stdout[-9000:])
if r.stderr:
    print(r.stderr[-2000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)