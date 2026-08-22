#pragma once

#include <stddef.h>
#include <stdint.h>

/** Fixed-width runtime implementation family. Automatic selects the best available family. */
typedef uint32_t uni_simd_backend_e;
enum {
    /** Selects the best implementation available at runtime. */
    UNI_SIMD_BACKEND_AUTOMATIC = 0,
    /** Portable scalar implementation. */
    UNI_SIMD_BACKEND_GENERIC = 1,
    /** x86 SSE2 implementation where available. */
    UNI_SIMD_BACKEND_X86_SSE2 = 2,
    /** x86 AVX2 implementation where available. */
    UNI_SIMD_BACKEND_X86_AVX2 = 3,
    /** x86 AVX2/FMA implementation where available. */
    UNI_SIMD_BACKEND_X86_AVX2_FMA = 4,
    /** x86 AVX-512 implementation where available. */
    UNI_SIMD_BACKEND_X86_AVX512 = 5,
    /** AArch64 NEON implementation where available. */
    UNI_SIMD_BACKEND_AARCH64_NEON = 6
};

/** Result returned by every public function. */
typedef uint32_t uni_simd_result_e;
enum {
    /** Operation completed successfully. */
    UNI_SIMD_RESULT_SUCCESS = 0,
    /** uni_simd_initialize() has not been called or the runtime was finalized. */
    UNI_SIMD_RESULT_NOT_INITIALIZED = 1,
    /** A pointer, enum value, parameter, or parameter combination is invalid. */
    UNI_SIMD_RESULT_INVALID_ARGUMENT = 2,
    /** A count, capacity, or supported transform size is invalid. */
    UNI_SIMD_RESULT_INVALID_SIZE = 3,
    /** Buffers overlap where the selected kernel forbids overlap. */
    UNI_SIMD_RESULT_OVERLAPPING_BUFFERS = 4,
    /** The requested backend is unavailable on this build or processor. */
    UNI_SIMD_RESULT_UNSUPPORTED_BACKEND = 5,
    /** An internal allocation failed. */
    UNI_SIMD_RESULT_OUT_OF_MEMORY = 6,
    /** State is incompatible with the operation or still alive during finalization. */
    UNI_SIMD_RESULT_INVALID_STATE = 7
};

/** Floating-point policy used while selecting a kernel implementation. */
typedef uint32_t uni_simd_math_mode_e;
enum {
    /** Uses the fastest selected implementation. */
    UNI_SIMD_MATH_FAST = 0,
    /** Uses deterministic generic floating-point implementations. */
    UNI_SIMD_MATH_DETERMINISTIC = 1
};

/** PFB frequency grid. */
typedef uint32_t uni_simd_pfb_grid_offset_e;
enum {
    /** Integer-spaced frequency bins. */
    UNI_SIMD_PFB_INTEGER_BINS = 0,
    /** Frequency bins shifted by one half-bin. */
    UNI_SIMD_PFB_HALF_BINS = 1
};

/** Identifier of a value in the parameter array passed to uni_simd_execute(). */
typedef uint32_t uni_simd_param_id_e;
enum {
    UNI_SIMD_PARAM_UNKNOWN = 0,
    /** U32 input: requested uni_simd_backend_e. */
    UNI_SIMD_PARAM_BACKEND = 1,
    /** U32 output: actual uni_simd_backend_e used by a successful call. */
    UNI_SIMD_PARAM_RESOLVED_BACKEND = 2,
    /** U32 input: uni_simd_math_mode_e. */
    UNI_SIMD_PARAM_MATH_MODE = 3,
    /** U32 input boolean: avoid high-power implementations during automatic selection. */
    UNI_SIMD_PARAM_PREFER_ENERGY_EFFICIENCY = 4,
    /** FLOAT32 input: quantizer multiplier. */
    UNI_SIMD_PARAM_SCALE = 5,
    /** FLOAT32 input: quantizer offset. */
    UNI_SIMD_PARAM_OFFSET = 6,
    /** FLOAT32 input: positive finite amplitude normalization factor. */
    UNI_SIMD_PARAM_NORMALIZATION_FACTOR = 7,
    /** FLOAT32 input: positive finite resolution bandwidth in hertz. */
    UNI_SIMD_PARAM_RBW_HZ = 8,
    /** CONST_POINTER input: borrowed array of float taps. */
    UNI_SIMD_PARAM_TAPS = 9,
    /** SIZE input: number of float taps. */
    UNI_SIMD_PARAM_TAP_COUNT = 10,
    /** FLOAT32 input: center tap for the symmetric dot product. */
    UNI_SIMD_PARAM_CENTER_TAP = 11,
    /** SIZE input: PFB transform size. */
    UNI_SIMD_PARAM_BIN_COUNT = 12,
    /** SIZE input: PFB decimation. */
    UNI_SIMD_PARAM_DECIMATION = 13,
    /** U32 input: uni_simd_pfb_grid_offset_e. */
    UNI_SIMD_PARAM_GRID_OFFSET = 14,
    /** CONST_POINTER input: borrowed array of int32_t logical bins. */
    UNI_SIMD_PARAM_LOGICAL_BINS = 15,
    /** SIZE input: number of logical bins. */
    UNI_SIMD_PARAM_LOGICAL_BIN_COUNT = 16,
    /** SIZE output: PFB samples produced or predicted per output buffer. */
    UNI_SIMD_PARAM_OUTPUT_COUNT = 17,
    /** U32 input boolean: query PFB output count without advancing state. */
    UNI_SIMD_PARAM_QUERY_OUTPUT_COUNT = 18,
    /** U32 input boolean: clear PFB streaming history before processing. */
    UNI_SIMD_PARAM_RESET = 19
};

/** Runtime type tag for uni_simd_param_t::value. */
typedef uint32_t uni_simd_param_type_e;
enum {
    /** Invalid placeholder type. */
    UNI_SIMD_PARAM_TYPE_UNKNOWN = 0,
    /** Value is stored in value.u32. */
    UNI_SIMD_PARAM_TYPE_U32 = 1,
    /** Value is stored in value.size. */
    UNI_SIMD_PARAM_TYPE_SIZE = 2,
    /** Value is stored in value.f32. */
    UNI_SIMD_PARAM_TYPE_FLOAT32 = 3,
    /** Value is stored in value.const_pointer and remains caller-owned. */
    UNI_SIMD_PARAM_TYPE_CONST_POINTER = 4
};

/** Number of adjacent floats used to store one interleaved complex sample: real, then imaginary. */
#define UNI_SIMD_CF32_COMPONENT_COUNT 2U

/** Borrowed read-only contiguous buffer; count is measured in kernel-specific elements. */
typedef struct uni_simd_const_buffer_t {
    const void* data;
    size_t count;
} uni_simd_const_buffer_t;

/** Borrowed writable contiguous buffer; count is its capacity in kernel-specific elements. */
typedef struct uni_simd_buffer_t {
    void* data;
    size_t count;
} uni_simd_buffer_t;

/** Array of output buffers used by the PFB channelizer. */
typedef struct uni_simd_buffer_array_t {
    uni_simd_buffer_t* buffers;
    size_t count;
} uni_simd_buffer_array_t;

/**
 * Mutable split-complex storage used by the batched in-place IFFT kernel.
 * stride is measured in float elements; zero selects transform_size. A zero
 * transform_count is a valid no-op and permits NULL data pointers.
 */
typedef struct uni_simd_split_cf32_t {
    float* real;
    float* imag;
    size_t descriptor_size;
    size_t transform_size;
    size_t transform_count;
    size_t stride;
} uni_simd_split_cf32_t;

/** Current value required in uni_simd_split_cf32_t::descriptor_size. */
#define UNI_SIMD_SPLIT_CF32_DESCRIPTOR_SIZE sizeof(uni_simd_split_cf32_t)

/** Typed parameter. IDs are unique within one call; duplicate IDs are invalid. */
typedef struct uni_simd_param_t {
    uni_simd_param_id_e param_id;
    uni_simd_param_type_e param_type;
    union {
        uint32_t u32;
        size_t size;
        float f32;
        const void* const_pointer;
    } value;
} uni_simd_param_t;

/** Opaque state. Only stateful kernels create instances of this type. */
typedef struct uni_simd_state_t uni_simd_state_t;
