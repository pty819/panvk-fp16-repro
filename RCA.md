# Root cause analysis: how "panvk fp16 bug" turned out to be a host memory bug

Timeline of the investigation that led to the (wrong) driver-bug conclusion,
the experiments that broke it open, and the actual root cause.

## The setup

The original reproducer (`repro-broken.c`) computes a small fp16 GEMM
(C[64][64] = A[64][8] x B[8][64], one 8x8-thread workgroup, f16vec2 packed
math) and full-verifies against a CPU reference. It reported:

```
panvk (Mali-G610):     FAIL: ~2718-2736 / 4096 wrong
llvmpipe (same SPIR-V): PASS: 0 / 4096 wrong
```

plus a handful of earlier observations that seemed to corroborate a driver bug:
an fp32 tiled GEMM "passing" (it later turned out to be 8-probe spot checks of
a 5-iteration loop, not a full verification), and small fp16 kernels failing
in several structural variants (packed accumulators, scalar half accumulators,
fp32 accumulators with fp16 multiplies).

## The experiment matrix that broke the case

New host (`diag.c`) + kernels (`e0`..`e5`), all full-verified, several
iterations each:

| kernel | precision | shape | panvk result |
|---|---|---|---|
| e0 | **fp32 only** | rolled loops, array accumulators | **FAIL ~2733** |
| e1 | scalar fp16 | rolled loops | FAIL, count varies 2514->1484->1284 across iterations |
| e2 | packed fp16 | phase-separated, + checksum probe | FAIL stable 2736; checksum probe also wrong 48/64 |
| e3 | packed fp16 | fused load->pack->FMA | FAIL stable ~2736, zero exact-zero outputs |
| e4 | packed fp16 | packed values round-tripped through LDS | FAIL stable ~2736 |
| e5 | packed fp16 | 4x4 tiles (low register pressure) | FAIL stable ~2736 |
| all of the above on llvmpipe | | | PASS |

The decisive observation: **e0 contains no fp16 at all and fails anyway.**
Whatever this is, it is not an fp16 bug.

Secondary observations that suddenly made sense later:

- 2736/4096 = 66.8% - suspiciously close to the fraction of reference values
  with |ref| > 1e-3, i.e. "everything is garbage, and 67% of it is far enough
  from the reference to count as wrong".
- The checksum probe D (an fp16 serial add chain over the packed A values)
  failed for exactly 48 of 64 threads - in groups of 8 threads that all read
  the same rows of A (threads with the same `lid.x`), i.e. **failure
  granularity followed A's memory layout, not the instruction schedule**.
- Wrong-element counts drifted downward across in-process iterations for some
  kernels (e1: 2514 -> 1484 -> 1284).
- Error "signatures" looked like partial sums: for many wrong elements,
  `got == ref - p_k` (one product term missing) for some k.

## The actual root cause

`vulkaninfo` on the device shows three memory types on one heap:

```
memtype 0: D---
memtype 1: DV-K    <- HOST_VISIBLE | HOST_CACHED, NOT HOST_COHERENT
memtype 2: DVC-    <- HOST_VISIBLE | HOST_COHERENT
```

Every host in the investigation picked **the first `HOST_VISIBLE` type** -
type 1, cached but non-coherent - and wrote A/B with plain `memcpy` on the
mapped pointer, then submitted the dispatch.

- CPU writes to A/B stayed dirty in cache; the GPU read **stale DRAM**
  (whatever the pages held before - mostly zeros, partly recycled data).
- That explains the partial-sum signatures (some cache lines had been written
  back naturally, others not), the A-layout failure granularity in the
  checksum probe, the ~67% wrong count (all-garbage output), and the
  run-to-run drift (progressive natural write-back).
- The C read-back direction has the same requirement
  (`vkInvalidateMappedMemoryRanges`); empirically the results were visible,
  but correctness there is not guaranteed either without it.

Control experiments on the GPU driver (both fix it completely):

1. Same host, same kernel, same non-coherent type, but with
   `vkFlushMappedMemoryRanges` after upload and
   `vkInvalidateMappedMemoryRanges` before read-back: **all kernels PASS**
   (e0-e5, g2_fp32, g3_fp32, and the original g5_fp16 - 0/4096 wrong,
   stable across 5 repeats).
2. Same host, no flush, but memory type selection changed to the
   `HOST_COHERENT` type: **all kernels PASS**.

And the pre-existing "fp32 was fine" datapoint dissolves: that host ran large
GEMMs 5 times in a loop and only spot-checked 8 elements afterwards - by then
dirty lines had been evicted naturally, so late iterations read fresh data.
The small single-dispatch tests (the ones that "found the bug") hit full
staleness.

## Why llvmpipe was a misleading control

llvmpipe executes the shader on the CPU. Its "device memory" is host memory
accessed through the same coherent caches the host just wrote. A host-side
coherence bug is invisible to this control - it is only a valid control for
shader/logic bugs. To catch this class of bug you need either a real second
GPU, a coherence-aware harness (always use coherent types or always flush), or
radically different memory layouts/sizes.

## Lessons

1. When a Vulkan host maps memory, "first HOST_VISIBLE type" is a footgun:
   pick `HOST_COHERENT` explicitly or manage flush/invalidate correctly.
2. A software-driver control does not exercise the memory path - it can only
   falsify shader-side hypotheses.
3. "Passing" spot checks on a loop of repeated dispatches are not evidence of
   correctness (cache pressure can make later iterations correct by accident).
4. Before blaming the driver, vary the parts of the system you did not write
   - here, printing `vkGetPhysicalDeviceMemoryProperties` would have taken
     one minute and ended the case immediately.

## Reproducing the postmortem

```sh
gcc -O2 -o repro repro.c -lvulkan && ./repro          # fixed host: PASS
gcc -O2 -o rb repro-broken.c -lvulkan && ./rb        # broken host: FAIL 2736/4096
```

Both binaries run the identical `gemm_fp16.spv`.
