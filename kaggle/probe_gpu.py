import subprocess


def sh(c):
    return subprocess.run(['bash', '-lc', c], capture_output=True, text=True)


print('=== GPU list ===')
print(sh('nvidia-smi -L').stdout)
print('=== GPU specs ===')
print(sh('nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version,clock.sm.max,clock.sm.cur,memory.used,temperature.gpu --format=csv').stdout)
print('=== CUDA toolchain ===')
print(sh('nvcc --version 2>&1 | tail -4').stdout)
print(sh('ls /usr/local/cuda/lib64/libcublas* 2>/dev/null | head -3').stdout)
print('=== Python / numpy ===')
print(sh('python3 -c "import sys, numpy; print(sys.version.split()[0], numpy.__version__)"').stdout)
print('=== system ===')
print(sh('free -g | head -2').stdout)
print(sh('df -h /kaggle/working | tail -1').stdout)
print(sh('python3 -c "import os; print(os.cpu_count())"').stdout)
print('=== network ===')
print(sh('curl -sI -m 15 https://huggingface.co | head -1').stdout)
print(sh('curl -sI -m 15 https://github.com | head -1').stdout)
