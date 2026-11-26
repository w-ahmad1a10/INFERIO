import os, traceback
import torch  # noqa
src = open("/kaggle/working/INFERIO/tools/bench_torch.py").read()
src = src.replace("    except Exception:\n        return 0",
                  "    except Exception:\n        traceback.print_exc()\n        return 0")
ns = {"__name__": "m", "__file__": "/kaggle/working/INFERIO/tools/bench_torch.py", "traceback": traceback}
exec(compile(src, "bench_torch.py", "exec"), ns)
x = torch.randn(1000, 1000, device="cuda")
print("clock:", ns["sample_clock_mhz"]())