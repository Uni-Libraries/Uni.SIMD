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

/** Identifier of a value configured with uni_simd_kernel_param_set(). */
typedef uint32_t uni_simd_param_id;
enum {
    UNI_SIMD_PARAM_UNKNOWN = 0,
    /** U32 input: requested uni_simd_backend_e. */
    UNI_SIMD_PARAM_BACKEND = 1,
    /** POINTER output: receives the uni_simd_backend_e used by a successful call. */
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
    /** POINTER output: receives the PFB samples produced or predicted per output buffer. */
    UNI_SIMD_PARAM_OUTPUT_COUNT = 17,
    /** U32 input boolean: query PFB output count without advancing state. */
    UNI_SIMD_PARAM_QUERY_OUTPUT_COUNT = 18,
    /** U32 input boolean: clear PFB streaming history before processing. */
    UNI_SIMD_PARAM_RESET = 19,
    /**
     * CONST_POINTER input: borrowed kernel-specific configuration descriptor.
     * Stateful kernels copy it when their state is created by first execution.
     */
    UNI_SIMD_PARAM_CONFIG = 20
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

#define UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT 4U

typedef struct uni_simd_qpsk_costas4_config_t {
    size_t descriptor_size;
    float alpha[UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT];
    float beta[UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT];
    float error_clip[UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT];
    float initial_phase[UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT];
    float initial_frequency[UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT];
} uni_simd_qpsk_costas4_config_t;

#define UNI_SIMD_QPSK_COSTAS4_CONFIG_DESCRIPTOR_SIZE sizeof(uni_simd_qpsk_costas4_config_t)

typedef struct uni_simd_qpsk_costas4_block_t {
    size_t descriptor_size;
    float* channels[UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT];
    size_t sample_count;
    float input_gain[UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT];
    float frequency_limit[UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT];
    uint32_t frequency_override_mask;
    float frequency_override[UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT];
} uni_simd_qpsk_costas4_block_t;

#define UNI_SIMD_QPSK_COSTAS4_BLOCK_DESCRIPTOR_SIZE sizeof(uni_simd_qpsk_costas4_block_t)

/**
 * Pipeline delay of the Costas loop in samples.
 *
 * The phase step decided from sample n rotates the phasor only after sample n + DELAY has been
 * de-rotated. Deciding a step (de-rotation, detector, loop filter, sin/cos) takes several times
 * longer than rotating the phasor, and with the delay the two overlap across samples instead of
 * following each other, so a sample costs about one phasor rotation. A carrier loop runs with a
 * bandwidth far below 1 / DELAY, where a few samples of extra loop delay cost a negligible
 * phase margin.
 */
#define UNI_SIMD_QPSK_COSTAS4_LOOP_DELAY 2U

/**
 * Loop state of the four-channel QPSK Costas kernel. `pending_*[k]` hold the steps decided but
 * not yet applied, oldest first: pending_*[0] rotates the phasor after the next sample.
 */
typedef struct uni_simd_qpsk_costas4_state_t {
    size_t descriptor_size;
    float phase[UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT];
    float phase_cos[UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT];
    float phase_sin[UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT];
    float frequency[UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT];
    float last_error[UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT];
    float pending_step[UNI_SIMD_QPSK_COSTAS4_LOOP_DELAY][UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT];
    float pending_step_cos[UNI_SIMD_QPSK_COSTAS4_LOOP_DELAY][UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT];
    float pending_step_sin[UNI_SIMD_QPSK_COSTAS4_LOOP_DELAY][UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT];
} uni_simd_qpsk_costas4_state_t;

#define UNI_SIMD_QPSK_COSTAS4_STATE_DESCRIPTOR_SIZE sizeof(uni_simd_qpsk_costas4_state_t)

/** Configuration for the allocation-free QPSK carrier-statistics analysis call. */
typedef struct uni_simd_qpsk_carrier_analyzer_config_t {
    size_t descriptor_size;
    /** A sample is valid for fourth-power statistics exactly when |x|^2 > magnitude_epsilon. */
    float magnitude_epsilon;
} uni_simd_qpsk_carrier_analyzer_config_t;

#define UNI_SIMD_QPSK_CARRIER_ANALYZER_CONFIG_DESCRIPTOR_SIZE \
    sizeof(uni_simd_qpsk_carrier_analyzer_config_t)

/** One borrowed block of interleaved CF32 input: real, then imaginary. */
typedef struct uni_simd_qpsk_carrier_analyzer_block_t {
    size_t descriptor_size;
    const float* samples;
    size_t sample_count;
} uni_simd_qpsk_carrier_analyzer_block_t;

#define UNI_SIMD_QPSK_CARRIER_ANALYZER_BLOCK_DESCRIPTOR_SIZE \
    sizeof(uni_simd_qpsk_carrier_analyzer_block_t)

/**
 * Unnormalized sums produced for one analyze call. Complex fields are stored
 * as {real, imaginary}. The analyzer kernel documentation defines the exact
 * equations and streaming-boundary behavior.
 */
typedef struct uni_simd_qpsk_carrier_analyzer_result_t {
    size_t descriptor_size;
    float fourth_sum[UNI_SIMD_CF32_COMPONENT_COUNT];
    float adjacent_fourth_sum[UNI_SIMD_CF32_COMPONENT_COUNT];
    float decision_sum[UNI_SIMD_CF32_COMPONENT_COUNT];
    float input_power;
    /** Sum of |x|^2 over samples counted by valid_fourth_count. */
    float valid_fourth_weight;
    size_t valid_fourth_count;
    size_t adjacent_fourth_count;
} uni_simd_qpsk_carrier_analyzer_result_t;

#define UNI_SIMD_QPSK_CARRIER_ANALYZER_RESULT_DESCRIPTOR_SIZE \
    sizeof(uni_simd_qpsk_carrier_analyzer_result_t)

#define UNI_SIMD_SYMBOL_SYNC4_CHANNEL_COUNT 4U

/** Largest samples-per-symbol value (omega_max) accepted by the four-channel symbol synchronizer. */
#define UNI_SIMD_SYMBOL_SYNC4_MAX_OMEGA 100.0

/** Interpolator used by the symbol synchronizer. Both use the four samples around the symbol. */
typedef uint32_t uni_simd_symbol_sync_interpolator_e;
enum {
    /** Cubic Lagrange (Farrow) interpolation through x[n-1], x[n], x[n+1], x[n+2]. */
    UNI_SIMD_SYMBOL_SYNC_INTERPOLATOR_CUBIC = 0,
    /** Linear interpolation between x[n] and x[n+1]. */
    UNI_SIMD_SYMBOL_SYNC_INTERPOLATOR_LINEAR = 1
};

/** Timing error detector of the symbol synchronizer. */
typedef uint32_t uni_simd_symbol_sync_ted_e;
enum {
    /** Gardner: e = Re{conj(mid) * (prev - cur)}, with mid half a symbol before cur. */
    UNI_SIMD_SYMBOL_SYNC_TED_GARDNER = 0,
    /** Decision-directed Mueller and Muller with QPSK decisions. */
    UNI_SIMD_SYMBOL_SYNC_TED_MUELLER_MULLER = 1
};

/**
 * Configuration of the four-channel symbol synchronizer. Every lane is an independent timing
 * loop; the per-lane arrays let channels with different rates share one kernel.
 *
 * Per symbol k at position t_k (in input samples) the loop computes the detector error e_k and
 *   omega_k = clamp(omega_{k-1} + beta * e_k, omega_min, omega_max)
 *   t_{k+1} = t_k + omega_k + alpha * e_k
 * The first symbol after reset has no error and advances by omega.
 *
 * With auto_ted set and ted == GARDNER, a lane switches to Mueller and Muller once the
 * exponential average of the squared normalised error, err_norm2 = (e / energy)^2 with
 * energy the power of the samples the detector used, stayed below mm_threshold for
 * mm_hold_symbols consecutive symbols (after at least mm_min_symbols symbols). With
 * auto_ted_fallback it switches back after fallback_hold_symbols symbols above
 * fallback_threshold. A zero hold count disables that switch. Each switch clears the
 * average and both run counters.
 */
typedef struct uni_simd_symbol_sync4_config_t {
    size_t descriptor_size;
    uni_simd_symbol_sync_interpolator_e interpolator;
    /** Detector every lane starts with. */
    uni_simd_symbol_sync_ted_e ted;
    /** Initial samples per symbol; 1 <= omega_min <= omega <= omega_max <= UNI_SIMD_SYMBOL_SYNC4_MAX_OMEGA. */
    double omega[UNI_SIMD_SYMBOL_SYNC4_CHANNEL_COUNT];
    double omega_min[UNI_SIMD_SYMBOL_SYNC4_CHANNEL_COUNT];
    double omega_max[UNI_SIMD_SYMBOL_SYNC4_CHANNEL_COUNT];
    /** Proportional (alpha) and integral (beta) loop gains, finite and non-negative. */
    double alpha[UNI_SIMD_SYMBOL_SYNC4_CHANNEL_COUNT];
    double beta[UNI_SIMD_SYMBOL_SYNC4_CHANNEL_COUNT];
    uint32_t auto_ted;
    uint32_t auto_ted_fallback;
    /** Smoothing factor of err_norm2 average, 0 < lock_ema_alpha < 1. */
    double lock_ema_alpha;
    uint64_t mm_min_symbols;
    uint32_t mm_hold_symbols;
    uint32_t fallback_hold_symbols;
    double mm_threshold;
    double fallback_threshold;
} uni_simd_symbol_sync4_config_t;

#define UNI_SIMD_SYMBOL_SYNC4_CONFIG_DESCRIPTOR_SIZE sizeof(uni_simd_symbol_sync4_config_t)

/**
 * One block per lane: interleaved CF32 input samples and a buffer for the interleaved CF32
 * symbols. Lanes may have different sample counts. The kernel keeps the few samples a symbol
 * near the end of a block still needs, so a stream may be split into blocks anywhere.
 */
typedef struct uni_simd_symbol_sync4_block_t {
    size_t descriptor_size;
    const float* input[UNI_SIMD_SYMBOL_SYNC4_CHANNEL_COUNT];
    size_t input_count[UNI_SIMD_SYMBOL_SYNC4_CHANNEL_COUNT];
    float* output[UNI_SIMD_SYMBOL_SYNC4_CHANNEL_COUNT];
    /**
     * Output capacity in symbols. input_count / omega_min + 2 is always enough while the loop
     * error is small; a lane whose buffer fills up drops the rest of its block and reports it
     * in uni_simd_symbol_sync4_result_t::truncated_mask.
     */
    size_t output_capacity[UNI_SIMD_SYMBOL_SYNC4_CHANNEL_COUNT];
} uni_simd_symbol_sync4_block_t;

#define UNI_SIMD_SYMBOL_SYNC4_BLOCK_DESCRIPTOR_SIZE sizeof(uni_simd_symbol_sync4_block_t)

/** Symbols produced by one call and the loop state after it. */
typedef struct uni_simd_symbol_sync4_result_t {
    size_t descriptor_size;
    size_t output_count[UNI_SIMD_SYMBOL_SYNC4_CHANNEL_COUNT];
    /** Bit per lane whose output buffer filled up before its input was consumed. */
    uint32_t truncated_mask;
    uni_simd_symbol_sync_ted_e ted[UNI_SIMD_SYMBOL_SYNC4_CHANNEL_COUNT];
    double omega[UNI_SIMD_SYMBOL_SYNC4_CHANNEL_COUNT];
    double err_norm2_ema[UNI_SIMD_SYMBOL_SYNC4_CHANNEL_COUNT];
    uint64_t symbols[UNI_SIMD_SYMBOL_SYNC4_CHANNEL_COUNT];
    uint64_t ted_switches[UNI_SIMD_SYMBOL_SYNC4_CHANNEL_COUNT];
} uni_simd_symbol_sync4_result_t;

#define UNI_SIMD_SYMBOL_SYNC4_RESULT_DESCRIPTOR_SIZE sizeof(uni_simd_symbol_sync4_result_t)

/** Parameter value. The field used by each ID is documented above. */
typedef union uni_simd_param_val {
    uint32_t u32;
    size_t size;
    float f32;
    const void* const_pointer;
    void* pointer;
} uni_simd_param_val;

/** One typed parameter assignment for uni_simd_kernel_param_set[_many](). */
typedef struct uni_simd_param_t {
    uni_simd_param_id id;
    uni_simd_param_val value;
} uni_simd_param_t;

/** Opaque configured kernel instance. */
typedef struct uni_simd_kernel_t uni_simd_kernel_t;
