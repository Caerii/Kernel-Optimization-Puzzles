import torch
import triton
import triton.language as tl
import torch.nn as nn
import time
import os
from unsloth.kernels.utils import fast_dequantize
from bitsandbytes.nn import Linear4bit
from transformers.activations import ACT2FN
from transformers import set_seed

# Environment settings
os.environ["UNSLOTH_NO_PATCHING"] = "1"
os.environ["TORCH_COMPILE_DEBUG"] = "disable"

# ---------------- Triton Kernel Implementation ----------------
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

    # Dequant tensor 1 (explicitly to float16)
    packed1 = tl.load(ptr1 + byte_offsets, mask=byte_mask, other=0)
    shift1 = (offsets & 1) * 4
    q4bit_1 = (packed1 >> shift1) & 0x0F
    signed1 = (q4bit_1.to(tl.float16) - 8.0)
    scale1 = tl.load(abs_ptr1 + pid).to(tl.float16)
    tl.store(out1 + offsets, signed1 * scale1, mask=mask)

    # Dequant tensor 2 (explicitly to float16)
    packed2 = tl.load(ptr2 + byte_offsets, mask=byte_mask, other=0)
    shift2 = (offsets & 1) * 4
    q4bit_2 = (packed2 >> shift2) & 0x0F
    signed2 = (q4bit_2.to(tl.float16) - 8.0)
    scale2 = tl.load(abs_ptr2 + pid).to(tl.float16)
    tl.store(out2 + offsets, signed2 * scale2, mask=mask)

    # Dequant tensor 3 (explicitly to float16)
    packed3 = tl.load(ptr3 + byte_offsets, mask=byte_mask, other=0)
    shift3 = (offsets & 1) * 4
    q4bit_3 = (packed3 >> shift3) & 0x0F
    signed3 = (q4bit_3.to(tl.float16) - 8.0)
    scale3 = tl.load(abs_ptr3 + pid).to(tl.float16)
    tl.store(out3 + offsets, signed3 * scale3, mask=mask)


# ---------------- Custom Triton Wrapper ----------------

def custom_triton_dequantize(weight):
    nf4_tensor = weight.weight
    absmax_tensor = weight.weight.quant_state.absmax
    num_elements = nf4_tensor.numel() * 2

    output = torch.empty(num_elements, dtype=torch.float16, device='cuda')
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)

    fused_nf4_dequant_kernel[grid](
        nf4_tensor,
        absmax_tensor,
        output,
        num_elements=num_elements,
        SCALE_BLOCK_SIZE=64  # Critical for correctness
    )

    return output.view(weight.out_features, weight.in_features)

# ---------------- Unsloth Reference ----------------

def unsloth_dequantize(weight, dtype=torch.float16):
    return fast_dequantize(weight.weight, weight.weight.quant_state).to(dtype)


# ---------------- Network Definition ----------------

class MLP(nn.Module):
    def __init__(self, hd, m, dtype=torch.float16):
        super().__init__()
        self.gate_proj = Linear4bit(hd, m, bias=None, compute_dtype=dtype, quant_type="nf4").cuda()
        self.up_proj   = Linear4bit(hd, m, bias=None, compute_dtype=dtype, quant_type="nf4").cuda()
        self.down_proj = Linear4bit(m, hd, bias=None, compute_dtype=dtype, quant_type="nf4").cuda()
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# ---------------- Testing Utilities ----------------

def mlp_forward(X, mlp, dequantize_fx):
    up   = X @ dequantize_fx(mlp.up_proj).t()
    gate = X @ dequantize_fx(mlp.gate_proj).t()
    h = mlp.act_fn(gate) * up
    down = h @ dequantize_fx(mlp.down_proj).t()
    return down

def assert_same(x, y, dt):
    assert x.dtype == dt, f"Dtype mismatch {x.dtype} vs {dt}"
    torch.testing.assert_close(x, y, atol=1e-2, rtol=1e-2)

# ---------------- Correctness Check for bitsandbytes ----------------

def assert_correct_bnb(weight, dtype):
    assert(weight.weight.dtype == torch.uint8)
    assert(weight.weight.quant_state.dtype == dtype)
    assert(weight.weight.quant_state.absmax.dtype == torch.uint8)
    assert(weight.weight.quant_state.code.dtype == torch.float32)
    assert(weight.weight.quant_state.offset.dtype == torch.float32)
    assert(weight.weight.quant_state.blocksize == 64)
    assert(weight.weight.quant_state.state2.absmax.dtype == torch.float32)
    assert(weight.weight.quant_state.state2.code.dtype == torch.float32)
    assert(weight.weight.quant_state.state2.blocksize == 256)

# ---------------- Profiling Function ----------------

def profile_dequantization(dequantize_fx, method_name, repetitions=1000):
    set_seed(42)
    test_shapes = [
        (2, 3333, 2048, 8192, torch.float16),
        (5, 777, 1024, 4096, torch.bfloat16),
        (3, 2048, 4096, 14336, torch.bfloat16),
    ]

    total_elapsed = 0
    print(f"\n🚩 Benchmarking: {method_name}")

    for bsz, qlen, hd, m, dt in test_shapes:
        mlp = MLP(hd, m, dtype=dt).cuda()
        X = torch.randn((bsz, qlen, hd), dtype=dt, device='cuda')

        # Warm-up
        mlp(X)
        for _ in range(2):
            Y_ref = mlp(X)
            Y_test = mlp_forward(X, mlp, dequantize_fx)
            assert_same(Y_test, Y_ref, dt)
            assert_correct_bnb(mlp.up_proj, dt)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(repetitions):
            mlp_forward(X, mlp, dequantize_fx)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        avg_ms = (elapsed / repetitions) * 1000
        print(f"Shape {bsz, qlen, hd}: {avg_ms:.3f} ms/repetition")
        total_elapsed += elapsed

    print(f"✅ Total elapsed for {method_name}: {total_elapsed:.3f} sec\n")
    return total_elapsed

# ---------------- Main Benchmark ----------------

if __name__ == "__main__":
    unsloth_time = profile_dequantization(unsloth_dequantize, "Unsloth Kernel")
    triton_time = profile_dequantization(custom_triton_dequantize, "Custom Triton Kernel")

    speedup = unsloth_time / triton_time
    print(f"🔥 Triton kernel speedup over Unsloth: {speedup:.2f}x")
    if speedup >= 1.15:
        print("✅ Achieved required speedup (≥1.15x).")
    else:
        print("⚠️ Did not meet required speedup (≥1.15x). Consider further optimizations.")
