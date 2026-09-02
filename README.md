# Uni.SIMD

Uni.SIMD is a signal-processing library with a small C-compatible ABI and
runtime selection of scalar, SSE2, AVX2, AVX2/FMA, AVX-512, and AArch64 NEON
implementations. Consumer source code only includes the C header
`<uni/simd/uni_simd.h>`. The implementation uses C++23, so static-library
consumers must link the platform C++ runtime; the exported CMake target handles
that requirement.

## Build

```sh
git submodule update --init --recursive
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Tests and benchmarks default to enabled for a top-level build and disabled when
the project is included as a subdirectory. Use `UNI_SIMD_BUILD_TESTS`,
`UNI_SIMD_BUILD_BENCHMARKS`, and `UNI_SIMD_ENABLE_SANITIZERS` to override this.
CPU topology and OS-usable ISA features are provided by the vendored
`Uni.SysInfo` submodule.
`BUILD_SHARED_LIBS=ON` builds a shared library whose public ABI contains only
`uni_simd_initialize`, `uni_simd_finalize`, `uni_simd_kernel_create`,
`uni_simd_kernel_param_set`, `uni_simd_kernel_param_set_many`,
`uni_simd_kernel_reset`, `uni_simd_kernel_execute`, and `uni_simd_kernel_free`.

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

uni_simd_kernel_t* kernel =
    uni_simd_kernel_create(UNI_SIMD_KERNEL_PACK_BITS_LSB_U8);
if (kernel == NULL) {
    return 1;
}

uni_simd_result_e result =
    uni_simd_kernel_execute(kernel, &input, &output);
uni_simd_kernel_free(kernel);

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

Parameters are stored in a kernel instance with `uni_simd_kernel_param_set()`.
Each `uni_simd_param_t` contains an ID and value; the ID determines which value
field is used. `uni_simd_kernel_param_set_many()` atomically applies a batch and
leaves the previous configuration unchanged if any item is invalid.
Common optional parameters select a backend and math mode or report the resolved
backend through `value.pointer`. Kernel-specific parameters carry values such as
scale, taps, normalization, and PFB configuration.

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

uni_simd_kernel_t* kernel =
    uni_simd_kernel_create(UNI_SIMD_KERNEL_IFFT_SPLIT_CF32);
uni_simd_result_e result =
    uni_simd_kernel_execute(kernel, NULL, &transforms);
uni_simd_kernel_free(kernel);
```

Supported transform sizes are 4, 8, 16, and 32. `stride` is measured in float
elements and must be at least the transform size; zero means tightly packed.
A zero transform count is a valid no-op. The internal backend entry is batch-aware,
so one dispatch handles the complete descriptor. AVX2/FMA and AArch64 NEON use
shared radix-2 SIMD stages for 16/32-point transforms; the measured faster
radix-decomposition is retained for AVX2/FMA IFFT-8. IFFT-4 remains generic because
SIMD setup and permutation overhead is larger than the transform itself.

## Kernel Lifetime

Every operation uses an opaque `uni_simd_kernel_t`. Primitive and IFFT instances
hold configuration only. PFB, Costas, and carrier-analyzer instances additionally
own streaming state.

Configure PFB through `uni_simd_kernel_param_set()` before its first execution.
The first execution validates and copies the configuration and creates streaming
state. Later calls process blocks or query the next output count;
`uni_simd_kernel_reset()` clears PFB history directly. The legacy one-shot RESET
parameter remains accepted.
Costas and carrier-analyzer configuration is supplied through `UNI_SIMD_PARAM_CONFIG`
and copied on first execution. `uni_simd_kernel_reset()` restores the Costas initial
state or clears analyzer adjacency state. Creation parameters cannot be changed after
streaming state exists.
`uni_simd_kernel_free()` accepts NULL as a successful no-op, and
`uni_simd_finalize()` refuses to finalize while any kernel instance remains alive.

PFB supports 4, 8, 16, and 32 bins. Decimation may be any value from one through
the bin count. A configuration accepts at most 1025 finite real taps and up to eight
unique logical bins in `[-M/2, M/2-1]`. Taps are newest-first, initial history is
zero, and the first output is emitted for input sample zero. There is no
end-of-stream flush operation.

AVX2/FMA and AArch64 NEON PFB implementations cover every supported bin count,
decimation, tap count, grid offset, and output selection. An explicit AVX-512
request uses AVX-512 for 32-bin transforms and multi-output 8-bin transforms,
falling back to AVX2/FMA otherwise; automatic dispatch keeps AVX2/FMA because
AVX-512 is not consistently faster across core classes. SIMD code is specialized
only by vector/FFT width.
The mirrored history ring guarantees enough overwrite headroom for four output
hops to share coefficient loads. A single selected channel uses a direct SIMD
transform; multiple channels use one batched backend IFFT call for all queued
hops. Optimized FIR reductions may round differently for different block
fragmentation; deterministic math mode uses the generic path when bit-stable
execution is required.

For bin `b`, grid offset `delta` (`0` or `0.5`), decimation `D`, and hop `h`,
the channelizer computes the following unscaled output, with samples before the
start of the stream treated as zero:

```text
y_b[h] = sum_k taps[k] * input[h*D-k]
         * exp(-i*2*pi*(b+delta)*(h*D-k)/M)
```

Input and output use interleaved `{real, imaginary}` floats and are processed
directly without internal block-sized conversion buffers. Configuration and
coefficient tables are copied when streaming state is created; steady-state
processing does not allocate. A failed processing call does not reset or advance
state. Kernel instances have no internal operation lock: execute, parameter
updates, reset, and free must not overlap for the same instance. Create another
instance for concurrent work; separate instances may run concurrently.

## Benchmarks

```sh
./build/src_benchmark/uni_simd_benchmark
./build/src_benchmark/uni_simd_benchmark --thorough
./build/src_benchmark/uni_simd_benchmark --kernel pfb
./build/src_benchmark/uni_simd_benchmark --kernel pfb_cf32_8_four --backend avx2-fma
```

The default quick run uses three iterations, one warmup batch, and 1 ms samples;
all values can be overridden with command-line options. The benchmark requires
an optimized build, validates every unique available implementation against
generic output before timing, includes PFB and IFFT table entries, and runs once
per detected core class with thread affinity. IFFT entries use batched dispatch;
PFB bandwidth includes only caller-visible payload because processing has no
conversion scratch.
