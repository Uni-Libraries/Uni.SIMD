# Uni.SIMD

Uni.SIMD is a signal-processing library with a small C-compatible ABI and
runtime selection of scalar, SSE2, AVX2, AVX2/FMA, AVX-512, and AArch64 NEON
implementations. Consumer source code only includes the C header
`<uni/simd/uni_simd.h>`. The implementation uses C++23, so static-library
consumers must link the platform C++ runtime; the exported CMake target handles
that requirement.

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Tests and benchmarks default to enabled for a top-level build and disabled when
the project is included as a subdirectory. Use `UNI_SIMD_BUILD_TESTS`,
`UNI_SIMD_BUILD_BENCHMARKS`, and `UNI_SIMD_ENABLE_SANITIZERS` to override this.
`BUILD_SHARED_LIBS=ON` builds a shared library whose public ABI contains only
`uni_simd_initialize`, `uni_simd_finalize`, `uni_simd_execute`, and
`uni_simd_state_free`.

CMake generates `uni_simd_export.h` and `uni_simd_version.h` in the build tree.
The installed package exports the `Uni::SIMD` target and installs those generated
headers together with the three source headers.

ISA source files are classified centrally by suffix. New implementations use
`_generic`, `_sse2`, `_avx2`, `_avx2_fma`, `_avx512f`, `_avx512bw`, or `_neon`;
`_common` denotes baseline code. CMake rejects unregistered suffixes and applies
the corresponding compiler flags automatically.

## API

```c
#include <uni/simd/uni_simd.h>

uint8_t input_data[16] = {1, 0, 1, 1, 0, 0, 1, 0,
                          0, 1, 1, 0, 1, 0, 0, 1};
uint8_t output_data[2];
uni_simd_const_buffer_t input = {input_data, 16};
uni_simd_buffer_t output = {output_data, 2};

if (uni_simd_initialize() != UNI_SIMD_RESULT_SUCCESS) {
    return 1;
}

uni_simd_result_e result = uni_simd_execute(
    UNI_SIMD_KERNEL_PACK_BITS_LSB_U8,
    &input, &output, NULL, 0, NULL);

if (uni_simd_finalize() != UNI_SIMD_RESULT_SUCCESS) {
    return 1;
}
```

`input` and `output` are operation-specific pointers. Most kernels use
`uni_simd_const_buffer_t` and `uni_simd_buffer_t`; their `count` fields are
element counts. Complex buffers use flat `float` storage in
`{real0, imaginary0, real1, imaginary1, ...}` order. IFFT uses
`uni_simd_split_cf32_t`, and PFB uses a regular input buffer plus
`uni_simd_buffer_array_t`. The kernel documentation in
`uni_simd_kernels.h` defines each element type and required parameter.

Parameters are typed `uni_simd_param_t` entries. Common optional parameters
select a backend and math mode or report the resolved backend. Kernel-specific
parameters carry values such as scale, taps, normalization, and PFB
configuration. Duplicate IDs and mismatched runtime types are rejected.

### Batched IFFT

IFFT is an in-place, unscaled, positive-exponent transform over split-complex
`float` storage. One call can process a contiguous or strided batch, avoiding
runtime locking, parameter parsing, and dispatch for every small transform:

```c
float real[16] = {1.0f};
float imag[16] = {0.0f};
uni_simd_split_cf32_t transforms = {
    .real = real,
    .imag = imag,
    .descriptor_size = UNI_SIMD_SPLIT_CF32_DESCRIPTOR_SIZE,
    .transform_size = 8,
    .transform_count = 2,
    .stride = 8,
};

uni_simd_result_e result = uni_simd_execute(
    UNI_SIMD_KERNEL_IFFT_SPLIT_CF32,
    NULL, &transforms, NULL, 0, NULL);
```

Supported transform sizes are 4, 8, 16, and 32. `stride` is measured in float
elements and must be at least the transform size; zero means tightly packed.
A zero transform count is a valid no-op. The internal backend entry is batch-aware,
so one dispatch handles the complete descriptor. AVX2/FMA and AArch64 NEON use
shared radix-2 SIMD stages for 16/32-point transforms; the measured faster
radix-decomposition is retained for AVX2/FMA IFFT-8. IFFT-4 remains generic because
SIMD setup and permutation overhead is larger than the transform itself.

## State

All primitive and IFFT kernels are stateless. The streaming PFB channelizer is
the only stateful kernel and receives `uni_simd_state_t**`.

The first PFB call passes configuration parameters and a pointer to a NULL state;
the library validates and copies the configuration and returns an opaque state.
Later calls process blocks, query the next output count, or reset history.
`uni_simd_state_free()` releases the opaque state and accepts NULL as a no-op.
`uni_simd_finalize()` refuses to finalize while any state remains alive.

PFB supports 4, 8, 16, and 32 bins. Decimation is a nonzero divisor of the bin
count. A configuration accepts at most 1025 finite real taps and up to eight
unique logical bins in `[-M/2, M/2-1]`. Taps are newest-first, initial history is
zero, and the first output is emitted for input sample zero. There is no
end-of-stream flush operation.

AVX2/FMA and AArch64 NEON PFB implementations cover every supported bin count,
decimation, tap count, grid offset, and output selection. SIMD code is specialized
only by vector/FFT width; it has no hard-coded coefficient or channel profile.
Up to four output hops share coefficient loads when the mirrored history ring has
enough overwrite headroom. A single selected channel uses a direct SIMD transform;
multiple channels use one batched backend IFFT call for all queued hops. Optimized FIR reductions may
round differently for different block fragmentation; deterministic math mode uses
the generic path when bit-stable execution is required.

For bin `b`, grid offset `delta` (`0` or `0.5`), decimation `D`, and hop `h`,
the channelizer computes the following unscaled output, with samples before the
start of the stream treated as zero:

```text
y_b[h] = sum_k taps[k] * input[h*D-k]
         * exp(-i*2*pi*(b+delta)*(h*D-k)/M)
```

Input and output use interleaved `{real, imaginary}` floats and are processed
directly without internal block-sized conversion buffers. Configuration and
coefficient tables are copied at state creation; steady-state processing does
not allocate. A failed processing call does not reset or advance state. A state
is single-stream and must not be used concurrently, while separate states and
stateless IFFT calls may run concurrently.

## Benchmarks

```sh
./build/src_benchmark/uni_simd_benchmark
./build/src_benchmark/uni_simd_benchmark --thorough
```

The default quick run uses three iterations, one warmup batch, and 1 ms samples;
all values can be overridden with command-line options. The benchmark requires
an optimized build, validates every unique available implementation against
generic output before timing, includes PFB and IFFT table entries, and runs once
per detected core class with thread affinity. IFFT entries use batched dispatch;
PFB bandwidth includes only caller-visible payload because processing has no
conversion scratch.
