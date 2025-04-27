import torch
import triton
import triton.language as tl
import torch.nn as nn
from unsloth.kernels.utils import fast_dequantize
from bitsandbytes.nn import Linear4bit
from transformers.activations import ACT2FN
from transformers import set_seed
import time
import os

# Environment optimizations
os.environ["UNSLOTH_NO_PATCHING"] = "1"
os.environ["TORCH_COMPILE_DEBUG"] = "disable"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# -------------------- Triton Kernel Optimized --------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
    ],
    key=['num_elements'],
)
@triton.jit
def optimized_nf4_to_fp16_kernel(
    nf4_ptr, absmax_ptr, out_ptr, num_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Load packed nf4 bytes (efficient vectorized loads)
    byte_offsets = offsets // 2
    packed_bytes = tl.load(nf4_ptr + byte_offsets, mask=(byte_offsets < (num_elements + 1)//2), other=0)

    # Extract 4-bit values efficiently
    shift_amount = (offsets & 1) * 4
    quant_4bit = (packed_bytes >> shift_amount) & 0x0F

    # Convert to signed and float16 directly
    quantized_signed = quant_4bit.to(tl.float16) - 8.0

    # Efficient loading of scales
    scale_idx = offsets // BLOCK_SIZE
    block_scale = tl.load(absmax_ptr + scale_idx, mask=mask)

    # Dequantize
    dequantized = quantized_signed * block_scale

    # Store back
    tl.store(out_ptr + offsets, dequantized, mask=mask)

# -------------------- Triton Kernel Wrapper --------------------

def optimized_fast_nf4_dequantize(weight):
    nf4_tensor = weight.weight
    absmax_tensor = weight.weight.quant_state.absmax
    num_elements = nf4_tensor.numel() * 2
    output = torch.empty(num_elements, dtype=torch.float16, device='cuda').contiguous()
    BLOCK_SIZE = 256  # Initial guess, autotune handles optimal choice
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)

    optimized_nf4_to_fp16_kernel[grid](
        nf4_tensor, absmax_tensor, output, num_elements
    )
    return output.view(weight.out_features, weight.in_features)

# -------------------- Unsloth Baseline --------------------

def unsloth_dequantize(weight):
    return fast_dequantize(weight.weight, weight.weight.quant_state)

# -------------------- MLP Class --------------------

class MLP(nn.Module):
    def __init__(self, hd, m, dtype=torch.float16):
        super().__init__()
        self.gate_proj = Linear4bit(hd, m, bias=None, compute_dtype=dtype, compress_statistics=True, quant_type="nf4").cuda()
        self.up_proj   = Linear4bit(hd, m, bias=None, compute_dtype=dtype, compress_statistics=True, quant_type="nf4").cuda()
        self.down_proj = Linear4bit(m, hd, bias=None, compute_dtype=dtype, compress_statistics=True, quant_type="nf4").cuda()
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def mlp_dequantize(mlp, fx):
    a = fx(mlp.up_proj); torch.cuda.synchronize()
    b = fx(mlp.gate_proj); torch.cuda.synchronize()
    c = fx(mlp.down_proj); torch.cuda.synchronize()
    return a, b, c

# -------------------- Benchmark Function --------------------

def profile_method(method, shape, repetitions=500):
    bsz, qlen, hd = shape
    m = hd * 4
    set_seed(123)
    torch.set_default_dtype(torch.float32)
    mlp = MLP(hd=hd, m=m, dtype=torch.float16)
    X = torch.randn((bsz, qlen, hd), device="cuda", dtype=torch.float16)

    # Warm-up runs
    for _ in range(2):
        mlp_dequantize(mlp, method)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repetitions):
        mlp_dequantize(mlp, method)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    return (elapsed / repetitions) * 1000  # ms per repetition

# -------------------- Main Benchmarking --------------------

def main():
    shapes = [
        (2, 1024, 1024),
        (4, 2048, 4096),
        (8, 4096, 8192),
        (16, 512, 2048),
        (32, 256, 1024),
    ]

    print("\n🚩 NF4 Dequantization Kernel Benchmark 🚩\n")
    print(f"{'Shape':<20}{'Unsloth(ms)':<15}{'Custom Triton(ms)':<20}{'Speedup':<10}")
    print("-" * 65)

    for shape in shapes:
        unsloth_time = profile_method(unsloth_dequantize, shape)
        custom_time = profile_method(optimized_fast_nf4_dequantize, shape)
        speedup = unsloth_time / custom_time
        print(f"{str(shape):<20}{unsloth_time:<15.4f}{custom_time:<20.4f}{speedup:.2f}x")

    print("\n🔥 Benchmark Complete. Ensure speedup ≥ 1.15x for all shapes.\n")

if __name__ == "__main__":
    main()
