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

## Benchmarks

```sh
./build/src_benchmark/uni_simd_benchmark
./build/src_benchmark/uni_simd_benchmark --thorough
```

The default quick run uses three iterations, one warmup batch, and 1 ms samples;
all values can be overridden with command-line options. The benchmark requires
an optimized build, validates every unique available implementation against
generic output before timing, includes PFB and IFFT table entries, and runs once
per detected core class with thread affinity.
