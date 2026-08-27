# panvk-fp16-repro — RETRACTED: not a driver bug, a host-side memory bug

**TL;DR: this repo originally claimed a panvk packed-fp16 correctness bug. That
was wrong — do not file it upstream. The wrong results were caused by my own
host program picking a non-coherent memory type and never calling
`vkFlushMappedMemoryRanges` / `vkInvalidateMappedMemoryRanges`. With correct
memory management the exact same shader passes on panvk. The repo is kept as
a postmortem of a convincing-but-wrong debugging investigation.**

## What actually happened

The reproducer host allocated its buffers from the **first** `HOST_VISIBLE`
memory type. On this device (RK3588 / panvk) the memory types are:

```
memtype 0: heap 0 flags=D---            (device local)
memtype 1: heap 0 flags=DV-K             (HOST_VISIBLE | HOST_CACHED, NON-coherent)
memtype 2: heap 0 flags=DVC-             (HOST_VISIBLE | HOST_COHERENT)
```

Type 1 is cached but **not coherent**. The host wrote A/B via `memcpy` on the
mapped pointer — those writes sat in CPU cache while the GPU read **stale
DRAM**, producing garbage / partial-sum-looking outputs. Fixing either half
makes every test pass:

- selecting the `HOST_COHERENT` type (no flush needed), **or**
- keeping type 1 and calling `vkFlushMappedMemoryRanges` after upload +
  `vkInvalidateMappedMemoryRanges` before read-back

```
broken host (type 1, no flush):   FAIL: 2736/4096 wrong   (stable, x3)
fixed host (coherent + flush):    PASS:    0/4096 wrong   (stable, x3, and on llvmpipe)
```

`repro-broken.c` is the original host; `repro.c` is the fixed one. The shader
(`gemm_fp16.comp`, f16vec2 packed math) is identical in both — it was never
broken.

## Why the investigation went off the rails

The original "evidence" for a driver bug was:

1. **Same SPIR-V passes on llvmpipe but fails on panvk.** Useless as a
   control for *host* coherence bugs: llvmpipe runs on the CPU, where the
   "GPU" reads go through the same coherent caches the host just wrote.
2. **An fp32 variant of the kernel "passed" on panvk.** That variant had only
   ever been spot-checked at 8 probe points of large 512³ GEMMs run 5 times
   in a loop — big buffers and repeated dispatches gave dirty cache lines time
   to be written back naturally, so late iterations read fresh data. Small
   single-dispatch tests (the failing ones) hit full staleness.
3. **Run-to-run variance in the wrong-element count** (2718/2729/2736) looked
   like a hardware race. It was partial cache-line visibility of stale DRAM.

The clue that broke the case: a pure-fp32 control kernel written for the
root-cause experiments *also* failed, and failed with almost the same
wrong-element count (~66.9% = the fraction of reference values above the
1e-3 tolerance when the output is effectively garbage).

## Files

- `repro.c` — fixed host (coherent type selection + flush/invalidate): **PASS**
- `repro-broken.c` — original host (first HOST_VISIBLE type, no flush): **FAIL**
- `gemm_fp16.comp` / `.spv` / `.spvasm` — the compute shader (unchanged, correct)
- `RCA.md` — detailed root-cause analysis and the experiment matrix
- `panvk-fail.log` / `llvmpipe-pass.log` — outputs from the *broken* host, kept for the postmortem

## Environment

Orange Pi 5 (RK3588), Mali-G610 MC4, Mesa 26.0.8-1ubuntu0.3 (panvk, Vulkan
1.4.335), Ubuntu 26.04 arm64, Linux 7.1.8-edge-rockchip64.

## Takeaways

- On panvk (and UMA drivers generally), don't take the first `HOST_VISIBLE`
  memory type; check `HOST_COHERENT` or do explicit flush/invalidate.
- A CPU/software-rasterizer control can only rule out *shader* bugs — it
  cannot rule out host-side memory-management bugs.
- A spot-check that "passes" while a full verification fails isn't a pass.
