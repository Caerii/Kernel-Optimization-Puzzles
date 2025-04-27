#!/usr/bin/env python3
"""
max_batch_finder_verbose.py
---------------------------
Binary-search the largest batch that fits without CUDA-OOM, *with* detailed
progress prints so long iterations don’t appear stuck.
"""
import math, time, torch, gc
from verify_memory_efficient_linear import MemoryEfficientLinear, ce_head

# ------------------------------------------------------------------ config
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype   = torch.bfloat16
hidden  = 4_096
vocab   = 32_000          # pick a realistic value for your model
seqlen  = 4_096
chunk   = 256
bsz_max = 512

bar = "━" * 70
print(f"\n🔍  Searching max batch on {device}"
      f"\n{bar}\n"
      f"   hidden={hidden}  vocab={vocab:,}  seq={seqlen}  chunk={chunk}\n{bar}")

# ------------------------------------------------------------------ helper
def fits(batch: int, step: int) -> bool:
    """Run FWD+BWD once; return False only for CUDA OOM."""
    torch.cuda.empty_cache()
    try:
        t0 = time.perf_counter()
        size = batch * seqlen
        print(f"[{step:02}] Batch={batch:<4} | allocate ({size:,} × {hidden}) activations…",
              flush=True, end="")

        X      = torch.randn(size, hidden, device=device, dtype=dtype, requires_grad=True)
        labels = torch.randint(0, vocab, (size,), device=device)
        proj   = torch.nn.Linear(hidden, vocab, bias=False, device=device, dtype=dtype)

        print("  ✓", end="  ")
        t1 = time.perf_counter()
        print(f"FWD…", end="", flush=True)
        loss = MemoryEfficientLinear.apply(X, proj, labels, ce_head, chunk)
        t2 = time.perf_counter()
        print(f"  ✓  BWD…", end="", flush=True)
        loss.backward()
        t3 = time.perf_counter()

        print(f"  ✓  | times: alloc {t1-t0:.1f}s  fwd {t2-t1:.1f}s  bwd {t3-t2:.1f}s")
        return True

    except RuntimeError as err:
        if "out of memory" in str(err):
            torch.cuda.empty_cache()
            print("  ✗  OOM!")
            return False
        raise  # propagate other errors
    finally:
        # ensure tensors are freed before next iteration
        del X, labels, proj, loss
        gc.collect()
        torch.cuda.empty_cache()

# ------------------------------------------------------------------ search
lo, hi = 1, bsz_max
step   = 0
best   = 0
while lo <= hi:
    mid = (lo + hi) // 2
    step += 1
    ok = fits(mid, step)
    if ok:
        best = mid
        lo   = mid + 1
    else:
        hi   = mid - 1

print(f"\n✅  Largest batch that fits without OOM: {best}")
