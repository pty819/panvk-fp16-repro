# panvk packed-fp16 compute correctness reproducer

Self-contained reproducer for wrong results when a compute shader uses packed
fp16 (`f16vec2`) arithmetic on **panvk / Mali-G610 (RK3588)**.

- Same SPIR-V binary: **PASS on llvmpipe, FAIL on panvk**
- An fp32-only variant of the same kernel is **correct on panvk**
- No 16-bit storage is involved: A/B are plain fp32 SSBOs; only the shader
  arithmetic is fp16 (`GL_EXT_shader_explicit_arithmetic_types_float16`)

## Environment (tested)

| | |
|---|---|
| Board | Orange Pi 5 (Rockchip RK3588) |
| GPU | Mali-G610 MC4, `/dev/dri/renderD128` |
| Mesa | 26.0.8-1ubuntu0.3 (panvk, Vulkan 1.4.335, conformance 1.4.1.2) |
| OS / kernel | Ubuntu 26.04 (arm64), Linux 7.1.8-edge-rockchip64 |
| Loader / tools | vulkan loader 1.4.341, glslc, spirv-dis |

## Build & run

```sh
glslc gemm_fp16.comp -o gemm_fp16.spv      # or use the prebuilt .spv
gcc -O2 -o repro repro.c -lvulkan
./repro                                          # panvk -> FAIL
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json ./repro   # llvmpipe -> PASS
```

What it does: C[64][64] = A[64][8] x B[8][64], one 8x8-thread workgroup,
each thread computes an 8x8 output block with unrolled f16vec2 FMAs.
Deterministic LCG inputs; the C output is verified against a CPU reference
(inputs round-tripped through half, double accumulation, 1e-3 abs tolerance).

## Observed

panvk (see `panvk-fail.log`):

```
device : Mali-G610 MC4
driver : panvk Mesa 26.0.8-1ubuntu0.3
  C[ 0][ 0] expected -0.001485  got  0.000000
  C[ 0][ 1] expected -0.002418  got  0.000000
  ... (whole accumulator blocks come back zero)
FAIL: 2718/4096 wrong (abs tol 0.001)
```

llvmpipe (see `llvmpipe-pass.log`):

```
device : llvmpipe (LLVM 21.1.8, 128 bits)
driver : llvmpipe Mesa 26.0.8-1ubuntu0.3 (LLVM 21.1.8)
PASS: 0/4096 wrong (abs tol 0.001)
```

The wrong-element count varies slightly between runs (2718 vs 2729 vs 2736
observed), which suggests a scheduling/timing-dependent fault rather than a
pure codegen miscompile of a fixed instruction. Larger variants of the same
kernel (K=512..2048 tiled GEMM) fail the same way.

## Files

- `repro.c` — self-contained Vulkan host (no arguments)
- `gemm_fp16.comp` — the compute shader (fully unrolled register GEMM)
- `gemm_fp16.spv` — prebuilt SPIR-V (glslc, environment above)
- `gemm_fp16.spvasm` — `spirv-dis` listing for reference
- `panvk-fail.log`, `llvmpipe-pass.log` — sample outputs

## Notes

- The K=8 case means the hot loop executes exactly once, so this is not a
  loop-carried issue; it reproduces with a single dispatch of one workgroup.
- Tested variants that all fail on panvk: f16vec2 accumulators, scalar
  `float16_t` accumulators, and fp32 accumulators fed by packed fp16
  multiplies — pointing at the packed fp16 multiply/FMA lowering rather than
  the accumulator type.
- If you can run a development Mesa build, `PANVK_DEBUG`/`PAN_MESA_DEBUG`
  shader dumps of this kernel would be the next diagnostic step (the distro
  build here has debug printing disabled).
