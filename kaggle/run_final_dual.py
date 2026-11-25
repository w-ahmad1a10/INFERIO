import subprocess
import sys

cmd = (
    'cd /kaggle/working && rm -rf INFERIO && '
    'git clone -q https://github.com/w-ahmad1a10/INFERIO.git && cd INFERIO && '
    'make -C engine 2>&1 | grep -E "error" | head -10; '
    'echo "==== PARITY (GPU0, v0.8) ====" && '
    'CUDA_VISIBLE_DEVICES=0 python3 bench/parity_check.py '
    '--lib engine/build/libinferio.so --weights /kaggle/working/weights_gpt2 '
    '--golden /kaggle/working/golden --out temp/results/parity_v080.json && '
    'echo "==== DUAL RUN: v0.8 official on GPU0 | PyTorch baseline on GPU1 ====" && '
    '( CUDA_VISIBLE_DEVICES=0 INFERIO_COMMIT=$(git rev-parse --short HEAD) '
    'python3 bench/bench_prefill.py --all --gpu 0 --profile '
    '--note "v0.8.0: MMA attention with V tile staged in shared memory" '
    '> /kaggle/working/bench_v08.txt 2>&1 ) & W1=$!; '
    '( CUDA_VISIBLE_DEVICES=1 python3 tools/bench_torch.py '
    '--note "PyTorch eager GPT-2 fp16 reference (no torch.compile) on identical frozen inputs" '
    '> /kaggle/working/bench_torch.txt 2>&1 ) & W2=$!; '
    'wait $W1; E1=$?; wait $W2; E2=$?; '
    'echo "---- v0.8 official (GPU0) ----"; tail -14 /kaggle/working/bench_v08.txt; '
    'echo "---- PyTorch baseline (GPU1) ----"; tail -14 /kaggle/working/bench_torch.txt; '
    'echo "EXITS v08=$E1 torch=$E2"'
)
r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=1750)
print(r.stdout[-10000:])
if r.stderr:
    print(r.stderr[-3000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)
