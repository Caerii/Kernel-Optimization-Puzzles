import os
os.environ["TORCH_COMPILE_DEBUG"] = "disable"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["UNSLOTH_NO_PATCHING"] = "1"
os.environ["UNSLOTH_NO_COMPILE"] = "1"
os.environ["UNSLOTH_DISABLE_COMPILE"] = "1"

import torch
import torch.nn as nn
from unsloth.kernels.utils import fast_dequantize
from bitsandbytes.nn import Linear4bit
import time
from transformers import set_seed

def unsloth_dequantize(weight):
    return fast_dequantize(weight.weight, weight.weight.quant_state)

class MLP(nn.Module):
    def __init__(self, hd=2048, m=8192, dtype=torch.float16):
        super().__init__()
        self.gate_proj = Linear4bit(hd, m, bias=None, compute_dtype=dtype, compress_statistics=True, quant_type="nf4").cuda()
        self.up_proj = Linear4bit(hd, m, bias=None, compute_dtype=dtype, compress_statistics=True, quant_type="nf4").cuda()
        self.down_proj = Linear4bit(m, hd, bias=None, compute_dtype=dtype, compress_statistics=True, quant_type="nf4").cuda()
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def mlp_dequantize(X, mlp, fx):
    a = fx(mlp.up_proj).t(); torch.cuda.synchronize()
    b = fx(mlp.gate_proj).t(); torch.cuda.synchronize()
    c = fx(mlp.down_proj).t(); torch.cuda.synchronize()
    return a, b, c

def test_dequantize(dequantize_fx, repetitions=500):
    set_seed(123)
    dtype = torch.float16
    mlp = MLP(dtype=dtype)
    X = torch.randn((2, 1024, 2048), device="cuda", dtype=dtype)

    # Warmup
    mlp(X)
    for _ in range(5):
        mlp_dequantize(X, mlp, dequantize_fx)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repetitions):
        mlp_dequantize(X, mlp, dequantize_fx)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start

    print(f"✅ Avg execution: {elapsed_time/repetitions*1000:.4f} ms per repetition")

if __name__ == "__main__":
    test_dequantize(unsloth_dequantize)
