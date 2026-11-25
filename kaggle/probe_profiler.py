import shutil
import subprocess

print('ncu :', shutil.which('ncu'))
print('nsys:', shutil.which('nsys'))
print(subprocess.run(['bash', '-lc', 'ls /opt | grep -i -E "nvidia|nsight"; ls /usr/local/cuda/bin 2>/dev/null | grep -i -E "ncu|nsight|nvprof"'], capture_output=True, text=True).stdout)
print(subprocess.run(['bash', '-lc', 'apt-cache policy nsight-compute 2>/dev/null | head -5'], capture_output=True, text=True).stdout)
