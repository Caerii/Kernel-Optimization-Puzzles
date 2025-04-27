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
import itertools

# Environment optimization settings
os.environ["UNSLOTH_NO_PATCHING"] = "1"
os.environ["TORCH_COMPILE_DEBUG"] = "disable"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Custom Triton NF4 kernel with autotuning capability
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_stages=stages, num_warps=warps)
        for bs, stages, warps in itertools.product(
            [32, 64, 128, 256, 512, 1024],
            [1, 2, 3, 4],
            [1, 2, 4, 8],
        )
    ],
    key=["num_elements"],
)
@triton.jit
def nf4_to_fp16_kernel(nf4_ptr, absmax_ptr, out_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    byte_offsets = offsets // 2
    packed_bytes = tl.load(nf4_ptr + byte_offsets, mask=(byte_offsets < (num_elements + 1) // 2))

    shift_amount = (offsets & 1) * 4
    quant_4bit = (packed_bytes >> shift_amount) & 0x0F
    quantized_signed = quant_4bit - 8

    scale_idx = offsets // BLOCK_SIZE
    block_scale = tl.load(absmax_ptr + scale_idx)
    dequantized = quantized_signed * block_scale

    tl.store(out_ptr + offsets, dequantized, mask=mask)

def custom_fast_nf4_dequantize(weight):
    nf4_tensor = weight.weight
    absmax_tensor = weight.weight.quant_state.absmax
    num_elements = nf4_tensor.numel() * 2
    output = torch.empty(num_elements, dtype=torch.float16, device='cuda')
    grid = (triton.cdiv(num_elements, 256),)
    nf4_to_fp16_kernel[grid](nf4_tensor, absmax_tensor, output, num_elements)
    return output.view(weight.out_features, weight.in_features)

# Unsloth baseline
def unsloth_dequantize(weight):
    return fast_dequantize(weight.weight, weight.weight.quant_state)

class MLP(nn.Module):
    def __init__(self, hd, m, dtype=torch.float16):
        super().__init__()
        self.gate_proj = Linear4bit(hd, m, bias=None, compute_dtype=dtype, compress_statistics=True, quant_type="nf4").cuda()
        self.up_proj = Linear4bit(hd, m, bias=None, compute_dtype=dtype, compress_statistics=True, quant_type="nf4").cuda()
        self.down_proj = Linear4bit(m, hd, bias=None, compute_dtype=dtype, compress_statistics=True, quant_type="nf4").cuda()
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def mlp_dequantize(mlp, fx):
    a = fx(mlp.up_proj); torch.cuda.synchronize()
    b = fx(mlp.gate_proj); torch.cuda.synchronize()
    c = fx(mlp.down_proj); torch.cuda.synchronize()
    return a, b, c

def profile(method, shape, repetitions=500):
    bsz, qlen, hd, m, dtype = shape
    set_seed(1234)
    mlp = MLP(hd=hd, m=m, dtype=dtype)
    X = torch.randn((bsz, qlen, hd), device="cuda", dtype=dtype)

    # Warmup
    for _ in range(3):
        mlp_dequantize(mlp, method)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repetitions):
        mlp_dequantize(mlp, method)
    torch.cuda.synchronize()
    return (time.time() - start) / repetitions * 1000  # ms per repetition

def run_sweep():
    shapes = [
        (2, 512, 1024, 4096, torch.float16),
        (4, 1024, 2048, 8192, torch.float16),
        (8, 2048, 4096, 16384, torch.float16),
        (16, 256, 1024, 2048, torch.float16),
        (32, 128, 512, 1024, torch.float16),
        (64, 64, 256, 512, torch.float16),
        (128, 32, 128, 256, torch.float16),
        (256, 16, 64, 128, torch.float16),
    ]

    print(f"{'Shape':<20}{'Unsloth(ms)':<15}{'Triton(ms)':<15}{'Speedup':<10}")
    print("-" * 60)

    for shape in shapes:
        unsloth_time = profile(unsloth_dequantize, shape)
        triton_time = profile(custom_fast_nf4_dequantize, shape)
        speedup = unsloth_time / triton_time
        shape_str = str(shape[:-1])
        print(f"{shape_str:<20}{unsloth_time:<15.4f}{triton_time:<15.4f}{speedup:<10.2f}x")

    print("\n🔥 Benchmark Complete. Analyze shapes carefully for optimal kernel parameters.")

if __name__ == "__main__":
    run_sweep()
