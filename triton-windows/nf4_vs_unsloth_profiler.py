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
import pandas as pd

# Environment optimization settings
os.environ["UNSLOTH_NO_PATCHING"] = "1"
os.environ["TORCH_COMPILE_DEBUG"] = "disable"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Triton NF4 optimized kernel
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

    block_scale = tl.load(absmax_ptr + pid)
    dequantized = quantized_signed * block_scale

    tl.store(out_ptr + offsets, dequantized, mask=mask)

def custom_fast_nf4_dequantize(weight, BLOCK_SIZE=256):
    nf4_tensor = weight.weight
    absmax_tensor = weight.weight.quant_state.absmax
    num_elements = nf4_tensor.numel() * 2
    output = torch.empty(num_elements, dtype=torch.float16, device='cuda').contiguous()

    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    nf4_to_fp16_kernel[grid](nf4_tensor, absmax_tensor, output, num_elements, BLOCK_SIZE)
    return output.view(weight.out_features, weight.in_features)

# Reference Unsloth method
def unsloth_dequantize(weight):
    return fast_dequantize(weight.weight, weight.weight.quant_state)

# MLP definition
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
    a = fx(mlp.up_proj)
    b = fx(mlp.gate_proj)
    c = fx(mlp.down_proj)
    torch.cuda.synchronize()
    return a, b, c

# Unified profiling function
def profile_dequantization(method, repetitions, shapes, method_name):
    results = []

    for (bsz, qlen, hd, m, seed, dt) in shapes:
        set_seed(seed)
        mlp = MLP(hd=hd, m=m, dtype=dt)
        X = torch.randn((bsz, qlen, hd), device="cuda", dtype=dt)

        # Warmup
        for _ in range(2):
            mlp_dequantize(mlp, method)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(repetitions):
            mlp_dequantize(mlp, method)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        results.append({
            'Shape': f"{bsz}x{qlen}x{hd}",
            'Dim': f"{hd}->{m}->{hd}",
            'Dtype': str(dt).split('.')[-1],
            'Elapsed (s)': round(elapsed, 4),
            'Avg Time (ms)': round((elapsed / repetitions)*1000, 4),
            'Method': method_name
        })

    return results

# Main profiling
if __name__ == "__main__":
    repetitions = 1000
    test_shapes = [
        (2, 1024, 2048, 8192, 3407, torch.float16),
        (5, 2048, 4096, 16384, 3408, torch.bfloat16),
        (3, 512, 1024, 4096, 3409, torch.float16),
        (4, 4096, 8192, 32768, 3410, torch.bfloat16),
        (2, 1024, 4096, 14336, 3411, torch.float16),
    ]

    unsloth_results = profile_dequantization(unsloth_dequantize, repetitions, test_shapes, "Unsloth")

    block_sizes = [128, 256, 512, 1024, 2048]
    custom_results = []
    for BLOCK_SIZE in block_sizes:
        method = lambda w: custom_fast_nf4_dequantize(w, BLOCK_SIZE=BLOCK_SIZE)
        method_results = profile_dequantization(method, repetitions, test_shapes, f"Triton-{BLOCK_SIZE}")
        custom_results.extend(method_results)

    # Combine all results
    all_results = unsloth_results + custom_results

    # Convert results to DataFrame and print
    df = pd.DataFrame(all_results)
    print("\n📝 Benchmarking Results Summary:")
    print(df.pivot_table(
        index=['Shape', 'Dim', 'Dtype'],
        columns='Method',
        values='Avg Time (ms)'
    ).round(4))

    # Identify best BLOCK_SIZE
    best_speeds = []
    for BLOCK_SIZE in block_sizes:
        triton_times = df[df['Method'] == f"Triton-{BLOCK_SIZE}"]['Avg Time (ms)'].values
        unsloth_times = df[df['Method'] == 'Unsloth']['Avg Time (ms)'].values
        avg_speedup = (unsloth_times / triton_times).mean()
        best_speeds.append((BLOCK_SIZE, avg_speedup))

    best_BLOCK, best_speedup = max(best_speeds, key=lambda x: x[1])

    print(f"\n🚀 Best BLOCK_SIZE: {best_BLOCK} with avg speedup: {best_speedup:.2f}x")
    if best_speedup >= 1.15:
        print("✅ Goal of ≥1.15x speedup achieved!")
    else:
        print("⚠️ Goal not met, consider further kernel optimizations.")
