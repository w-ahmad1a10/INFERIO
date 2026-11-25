import subprocess
import sys

cmd = (
    'cd /kaggle/working/INFERIO && '
    'echo "==== ncu attn_mma_kernel @ B=64 (GPU1) ====" && '
    'CUDA_VISIBLE_DEVICES=1 ncu --target-processes all -k regex:attn_mma_kernel '
    '--launch-count 2 --set detailed -o /kaggle/working/ncu_attn_b64 '
    'python3 bench/bench_prefill.py --single --batch 64 '
    '--lib engine/build/libinferio.so --weights /kaggle/working/weights_gpt2 '
    '> /kaggle/working/ncu_run.txt 2>&1; '
    'tail -3 /kaggle/working/ncu_run.txt; '
    'echo "==== details ====" && '
    'ncu --import /kaggle/working/ncu_attn_b64.ncu-rep --page details 2>/dev/null | '
    'grep -E "attn_mma|Compute \(SM\)|Memory Throughput|DRAM Throughput|L1/TEX Hit|Achieved Occupancy|Registers Per Thread|Block Limit|Warp Cycles Per|Stall|Issue|SM Busy|tensor" | head -40'
)
r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=1700)
print(r.stdout[-9000:])
if r.stderr:
    print(r.stderr[-2000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)