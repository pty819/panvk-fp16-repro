# panvk: packed fp16 (f16vec2) compute shader gives wrong results on Mali-G610 (RK3588)

## Summary

A compute shader that performs packed fp16 arithmetic (`f16vec2` from
`GL_EXT_shader_explicit_arithmetic_types_float16`) produces wrong results on
panvk with Mali-G610: roughly two thirds of the output elements of the
reproducer come back `0.0` or wrong, with whole accumulator blocks dropped.
The **exact same SPIR-V binary passes on llvmpipe**, and an fp32-only variant
of the same kernel is **correct on panvk** — so this looks like a panvk /
Valhall compiler issue in the packed fp16 multiply/FMA path, not a
reproducer bug.

No 16-bit storage is involved: the input buffers are plain fp32 SSBOs and the
shader converts to `f16vec2` in-register. Only `shaderFloat16` is enabled.

## Environment

- Board: Orange Pi 5 (Rockchip RK3588), Mali-G610 MC4 (CSF firmware)
- Mesa: 26.0.8-1ubuntu0.3 (Ubuntu 26.04 arm64 package)
- panvk reports: Vulkan 1.4.335, conformance version 1.4.1.2
- Kernel: Linux 7.1.8-edge-rockchip64 (mainline-based)
- vulkan loader/tools: 1.4.341

I could not find an existing issue for this; apologies if it's a duplicate.
Mesa 26.1/26.2 release notes don't appear to mention a fix for this, but I
have only been able to test 26.0.8.

## How to reproduce

Attached tarball `panvk-fp16-repro.tar.gz` (self-contained: host source,
shader source, prebuilt SPIR-V, disassembly, logs).

```sh
glslc gemm_fp16.comp -o gemm_fp16.spv      # or use prebuilt .spv
gcc -O2 -o repro repro.c -lvulkan
./repro                                                       # panvk  -> FAIL
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json ./repro  # llvmpipe -> PASS
```

It computes `C[64][64] = A[64][8] × B[8][64]` with a single 8x8-thread
workgroup; each thread computes an 8x8 output tile via unrolled `f16vec2`
FMAs. With K=8 the k-loop runs exactly once, so this is not a loop-carried
issue. Inputs are deterministic (LCG); the CPU reference round-trips inputs
through half and accumulates in double (1e-3 absolute tolerance).

## Observed

panvk (`panvk-fail.log`):

```
device : Mali-G610 MC4
driver : panvk Mesa 26.0.8-1ubuntu0.3
  C[ 0][ 0] expected -0.001485  got  0.000000
  C[ 0][ 1] expected -0.002418  got  0.000000
  ...
FAIL: 2718/4096 wrong (abs tol 0.001)
```

llvmpipe (`llvmpipe-pass.log`):

```
device : llvmpipe (LLVM 21.1.8, 128 bits)
PASS: 0/4096 wrong (abs tol 0.001)
```

Noteworthy details:

- The wrong-element count varies slightly between runs (2718 / 2729 / 2736
  observed across runs), while the wrongly-read elements are largely the
  same. That suggests something scheduling/timing-dependent rather than a
  deterministic miscompile of fixed instructions.
- Variants tested that all fail on panvk: `f16vec2` accumulators, scalar
  `float16_t` accumulators, and fp32 accumulators fed by packed fp16
  multiplies — pointing at the packed fp16 multiply/FMA lowering rather than
  the accumulator type.
- Larger versions of the same kernel (tiled GEMM, K=512..2048) fail the same
  way.
- (Separate from this report, I also hit two `VK_DEVICE_LOST` hangs with
  other compute kernels on this stack — high-register-pressure shaders and a
  large GEMM dispatch. Happy to file separately with reproducers if useful.)

## Attachments

- `panvk-fp16-repro.tar.gz` — reproducer sources + prebuilt SPIR-V +
  `spirv-dis` listing + sample logs
