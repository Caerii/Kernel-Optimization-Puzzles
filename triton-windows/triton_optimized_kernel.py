import torch
import triton
import triton.language as tl
from unsloth.kernels.utils import fast_dequantize
from bitsandbytes.nn import Linear4bit
from transformers.activations import ACT2FN
from transformers import set_seed
import torch.nn as nn
import time
import os

# Environment settings
os.environ["UNSLOTH_NO_PATCHING"] = "1"
os.environ["TORCH_COMPILE_DEBUG"] = "disable"

# ---- Custom Triton Kernel (Fused without nested functions) ----
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['num_elements']
)
@triton.jit
def fused_nf4_dequant_kernel(
    ptr1, abs_ptr1, out1,
    ptr2, abs_ptr2, out2,
    ptr3, abs_ptr3, out3,
    num_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    byte_offsets = offsets // 2
    byte_mask = byte_offsets < (num_elements + 1) // 2

    # Dequant tensor 1
    packed1 = tl.load(ptr1 + byte_offsets, mask=byte_mask, other=0)
    shift1 = (offsets & 1) * 4
    q4bit_1 = (packed1 >> shift1) & 0x0F
    signed1 = q4bit_1 - 8
    scale1 = tl.load(abs_ptr1 + pid)
    tl.store(out1 + offsets, signed1 * scale1, mask=mask)

    # Dequant tensor 2
    packed2 = tl.load(ptr2 + byte_offsets, mask=byte_mask, other=0)
    shift2 = (offsets & 1) * 4
    q4bit_2 = (packed2 >> shift2) & 0x0F
    signed2 = q4bit_2 - 8
    scale2 = tl.load(abs_ptr2 + pid)
    tl.store(out2 + offsets, signed2 * scale2, mask=mask)

    # Dequant tensor 3
    packed3 = tl.load(ptr3 + byte_offsets, mask=byte_mask, other=0)
    shift3 = (offsets & 1) * 4
    q4bit_3 = (packed3 >> shift3) & 0x0F
    signed3 = q4bit_3 - 8
    scale3 = tl.load(abs_ptr3 + pid)
    tl.store(out3 + offsets, signed3 * scale3, mask=mask)

# ---- Wrapper to launch fused kernel ----
def fused_triton_dequantize(gate, up, down):
    num_elements = gate.weight.numel() * 2
    dtype, device = torch.float16, 'cuda'

    gate_out = torch.empty((gate.out_features, gate.in_features), dtype=dtype, device=device)
    up_out = torch.empty((up.out_features, up.in_features), dtype=dtype, device=device)
    down_out = torch.empty((down.out_features, down.in_features), dtype=dtype, device=device)

    grid = (triton.cdiv(num_elements, 256),)
    fused_nf4_dequant_kernel[grid](
        gate.weight, gate.weight.quant_state.absmax, gate_out,
        up.weight, up.weight.quant_state.absmax, up_out,
        down.weight, down.weight.quant_state.absmax, down_out,
        num_elements
    )
    return gate_out, up_out, down_out

# ---- UNSLOTH Reference Dequantization ----
def unsloth_dequantize(weight):
    return fast_dequantize(weight.weight, weight.weight.quant_state)

# ---- MLP Module Definition ----
class MLP(nn.Module):
    def __init__(self, hd, m, dtype=torch.float16):
        super().__init__()
        self.gate_proj = Linear4bit(hd, m, bias=None, compute_dtype=dtype, quant_type="nf4").cuda()
        self.up_proj = Linear4bit(hd, m, bias=None, compute_dtype=dtype, quant_type="nf4").cuda()
        self.down_proj = Linear4bit(m, hd, bias=None, compute_dtype=dtype, quant_type="nf4").cuda()
        self.act_fn = ACT2FN["silu"]

# ---- Benchmarking Helpers ----
def profile(method, mlp, reps=500):
    torch.cuda.synchronize()
    for _ in range(5): method(mlp)  # Warm-up
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(reps): method(mlp)
    torch.cuda.synchronize()
    return (time.time() - start) * 1000 / reps

# ---- Method Wrappers for Profiling ----
def unsloth_method(mlp):
    a = unsloth_dequantize(mlp.gate_proj)
    b = unsloth_dequantize(mlp.up_proj)
    c = unsloth_dequantize(mlp.down_proj)
    return a, b, c

def triton_method(mlp):
    return fused_triton_dequantize(mlp.gate_proj, mlp.up_proj, mlp.down_proj)

# ---- Main Benchmark Function ----
def main_benchmark():
    set_seed(42)
    shapes = [
        (2, 512, 1024, 4096),
        (4, 1024, 2048, 8192),
        (8, 2048, 4096, 16384),
        (16, 256, 1024, 2048),
        (32, 128, 512, 1024),
        (64, 64, 256, 512),
        (128, 32, 128, 256),
        (256, 16, 64, 128),
    ]

    print(f"{'Shape':<25}{'Unsloth(ms)':<15}{'Triton(ms)':<15}{'Speedup':<10}")
    print("-" * 65)

    for (bsz, seq, hd, m) in shapes:
        mlp = MLP(hd, m).cuda()

        time_unsloth = profile(unsloth_method, mlp)
        time_triton = profile(triton_method, mlp)
        speedup = time_unsloth / time_triton

        # Print the names of the columns too
        print("bsz, seq, hd, m, Unsloth(ms), Triton(ms), Speedup")
        print(f"{(bsz, seq, hd, m)!s:<25}{time_unsloth:<15.4f}{time_triton:<15.4f}{speedup:<10.2f}x")

    print("\n✅ Benchmarking complete. Ensure Triton speedup ≥ 1.15x across all sizes.")

if __name__ == "__main__":
    main_benchmark()
