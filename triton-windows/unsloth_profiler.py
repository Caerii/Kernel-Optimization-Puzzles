
import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers import set_seed
import time
import inspect
import os

os.environ["UNSLOTH_NO_PATCHING"] = "1"
os.environ["TORCH_COMPILE_DEBUG"] = "disable"
from unsloth.kernels.utils import fast_dequantize

# Basic helper functions
major_version, minor_version = torch.cuda.get_device_capability()
HAS_BFLOAT16 = (major_version >= 8)

def NAME(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    names = [var_name for var_name, var_val in callers_local_vars if var_val is var]
    return names[0] if names else ""

def assert_same(x, y, dtype):
    assert x.dtype == dtype
    torch.testing.assert_close(x, y, check_stride=True, atol=1e-2, rtol=1e-2)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Import required modules
from bitsandbytes.nn import Linear4bit

def unsloth_dequantize(weight):
    return fast_dequantize(weight.weight, weight.weight.quant_state)

def bnb_Linear4bit(hd, m, dtype=torch.float16):
    return Linear4bit(
        hd, m, bias=None,
        compute_dtype=dtype,
        compress_statistics=True,
        quant_type="nf4",
    )

def assert_correct_bnb(weight, dtype):
    assert weight.weight.dtype == torch.uint8
    assert weight.weight.quant_state.dtype == dtype
    assert weight.weight.quant_state.absmax.dtype == torch.uint8
    assert weight.weight.quant_state.code.dtype == torch.float32
    assert weight.weight.quant_state.offset.dtype == torch.float32
    assert weight.weight.quant_state.blocksize == 64
    assert weight.weight.quant_state.state2.absmax.dtype == torch.float32
    assert weight.weight.quant_state.state2.code.dtype == torch.float32
    assert weight.weight.quant_state.state2.blocksize == 256

class MLP(nn.Module):
    def __init__(self, hd=4096, m=14336, dtype=torch.float16):
        super().__init__()
        self.gate_proj = bnb_Linear4bit(hd, m, dtype=dtype).to("cuda")
        self.up_proj   = bnb_Linear4bit(hd, m, dtype=dtype).to("cuda")
        self.down_proj = bnb_Linear4bit(m, hd, dtype=dtype).to("cuda")
        self.gate_proj.weight.quant_state.dtype = dtype
        self.up_proj.weight.quant_state.dtype = dtype
        self.down_proj.weight.quant_state.dtype = dtype
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def mlp_forward(X, mlp, fx):
    up   = X @ fx(mlp.up_proj).t()
    gate = X @ fx(mlp.gate_proj).t()
    h = mlp.act_fn(gate) * up
    down = h @ fx(mlp.down_proj).t()
    return down

def mlp_dequantize(X, mlp, fx):
    a = fx(mlp.up_proj).t(); torch.cuda.synchronize()
    b = fx(mlp.gate_proj).t(); torch.cuda.synchronize()
    c = fx(mlp.down_proj).t(); torch.cuda.synchronize()
    return a, b, c

def test_dequantize(dequantize_fx, repetitions=1000):
    elapsed = 0
    options = [
        (2, 3333, 2048,  8192, 3407, torch.float16),
        (5,  777, 1024,  4096, 3409, torch.bfloat16),
        (3, 2048, 4096, 14336, 3408, torch.bfloat16),
    ]

    for (bsz, qlen, hd, m, seed, dt) in options:
        set_seed(seed)
        torch.set_default_dtype(torch.float32)
        mlp = MLP(hd=hd, m=m, dtype=dt)
        X = torch.randn((bsz, qlen, hd), device="cuda", dtype=dt)
        torch.cuda.synchronize()

        # Warmup
        for _ in range(2):
            assert_same(mlp_forward(X, mlp, dequantize_fx), mlp(X), dt)
            assert_correct_bnb(mlp.up_proj, dt)
            assert_correct_bnb(mlp.gate_proj, dt)
            assert_correct_bnb(mlp.down_proj, dt)
            a, b, c = mlp_dequantize(X, mlp, dequantize_fx)
            A, B, C = mlp_dequantize(X, mlp, unsloth_dequantize)
            assert_same(a, A, dt)
            assert_same(b, B, dt)
            assert_same(c, C, dt)

        # Benchmarking
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(repetitions):
            mlp_dequantize(X, mlp, dequantize_fx)
        torch.cuda.synchronize()
        elapsed_time = time.time() - start
        elapsed += elapsed_time
        print(f"Shape: {(bsz, qlen, hd)} | Elapsed: {elapsed_time:.4f} sec")

    print(f"\n🚀 Total Elapsed Time (all tests): {elapsed:.4f} seconds")
    print(f"✅ Avg per repetition: {(elapsed / repetitions)*1000:.4f} ms")

# Run profiling test for Unsloth
if __name__ == "__main__":
    print("🚩 Benchmarking Unsloth fast_dequantize:")
    test_dequantize(unsloth_dequantize)
