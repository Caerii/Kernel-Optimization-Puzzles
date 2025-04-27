import torch
import triton
import triton.language as tl

@triton.jit
def nf4_to_fp16_kernel(
    nf4_ptr, absmax_ptr, out_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    byte_offsets = offsets // 2
    byte_mask = byte_offsets < (num_elements + 1) // 2

    packed_bytes = tl.load(nf4_ptr + byte_offsets, mask=byte_mask)
    is_high = offsets % 2
    quant_4bit = tl.where(is_high == 1, packed_bytes >> 4, packed_bytes & 0x0F)

    quantized_signed = quant_4bit - 8
    scale = tl.load(absmax_ptr + (offsets // BLOCK_SIZE))
    dequantized = quantized_signed * scale

    tl.store(out_ptr + offsets, dequantized, mask=mask)

def fast_nf4_dequantize(nf4_tensor, absmax_tensor, BLOCK_SIZE=256):
    num_elements = nf4_tensor.numel() * 2
    output = torch.empty(num_elements, dtype=torch.float16, device='cuda')
    grid = lambda meta: (triton.cdiv(num_elements, BLOCK_SIZE), )

    nf4_to_fp16_kernel[grid](
        nf4_tensor, absmax_tensor, output, num_elements, BLOCK_SIZE
    )
    return output

def profile_kernel(repetitions=500, tensor_size=(1<<22)):
    nf4_tensor = torch.randint(0, 255, (tensor_size,), dtype=torch.uint8, device='cuda')
    absmax_tensor = torch.rand(tensor_size // 16, dtype=torch.float16, device='cuda')

    fast_nf4_dequantize(nf4_tensor, absmax_tensor)  # Warmup run

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(repetitions):
        fast_nf4_dequantize(nf4_tensor, absmax_tensor)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event) / repetitions

    num_bytes_processed = tensor_size * 0.5 * 2
    bandwidth = num_bytes_processed / (elapsed_time_ms / 1e3) / 1e9

    print(f"✅ Avg execution: {elapsed_time_ms:.4f} ms per run")
    print(f"🔥 Memory Bandwidth: {bandwidth:.2f} GB/s")

if __name__ == "__main__":
    print("🚩 Running Enhanced Kernel Profiler:")
    profile_kernel()
