#pragma once

#include <uni_simd_export.h>
#include <uni_simd_kernels.h>
#include <uni_simd_typedefs.h>
#include <uni_simd_version.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef UNI_SIMD_CALL
#if defined(_WIN32)
#define UNI_SIMD_CALL __cdecl
#else
#define UNI_SIMD_CALL
#endif
#endif

/**
 * Initializes the process-wide Uni.SIMD runtime.
 *
 * The function is thread-safe and idempotent. Calls to uni_simd_execute()
 * before successful initialization return UNI_SIMD_RESULT_NOT_INITIALIZED.
 */
UNI_SIMD_API uni_simd_result_e UNI_SIMD_CALL uni_simd_initialize(void);

/**
 * Finalizes the process-wide runtime.
 *
 * The function is thread-safe and idempotent. It returns
 * UNI_SIMD_RESULT_INVALID_STATE without finalizing while any opaque state is
 * alive. Release every state through uni_simd_state_free() first.
 */
UNI_SIMD_API uni_simd_result_e UNI_SIMD_CALL uni_simd_finalize(void);

/**
 * Releases state created by a stateful kernel.
 *
 * Passing NULL is allowed and has no effect. The state must not be used by
 * another thread while it is being released. The caller must discard its
 * pointer after this function returns.
 */
UNI_SIMD_API void UNI_SIMD_CALL uni_simd_state_free(uni_simd_state_t* state);

/**
 * Executes one kernel.
 *
 * Common optional parameters are U32 BACKEND (default AUTOMATIC), U32
 * MATH_MODE (default FAST), U32 PREFER_ENERGY_EFFICIENCY (0 or 1), and U32
 * RESOLVED_BACKEND. RESOLVED_BACKEND is an output parameter updated with the
 * implementation that actually ran. Parameter IDs must be unique and their
 * runtime types must match the declarations in uni_simd_typedefs.h. For PFB,
 * backend and math-selection parameters are accepted only while creating state.
 *
 * Stateless kernels ignore state and require it to be NULL or to point to a
 * NULL state. Their input/output descriptor types and kernel-specific
 * parameters are documented in uni_simd_kernels.h.
 *
 * PFB protocol:
 * - Creation: *state must be NULL. Required parameters are SIZE BIN_COUNT,
 *   SIZE DECIMATION, U32 GRID_OFFSET, CONST_POINTER TAPS, SIZE TAP_COUNT,
 *   CONST_POINTER LOGICAL_BINS, and SIZE LOGICAL_BIN_COUNT. Configuration is
 *   copied. input/output may be NULL to create without processing.
 * - Processing: input points to uni_simd_const_buffer_t whose data is a flat
 *   interleaved {real, imaginary} float array; output points to
 *   uni_simd_buffer_array_t with one equally formatted buffer per logical bin.
 *   Buffer counts are complex-sample counts. SIZE OUTPUT_COUNT is an optional
 *   output parameter receiving samples produced per output.
 * - Query: U32 QUERY_OUTPUT_COUNT=1 and a SIZE OUTPUT_COUNT parameter compute
 *   the next output count from input->count without advancing state; output is
 *   ignored and input->data may be NULL.
 * - Reset: U32 RESET=1 clears history before any processing in the same call.
 * - Destruction: release the opaque state with uni_simd_state_free().
 *
 * Buffers are borrowed for the duration of the call. Except for COPY_U8 and
 * documented exact in-place kernels, active input and output ranges must not
 * overlap. A stateful instance must not be used concurrently.
 */
UNI_SIMD_API uni_simd_result_e UNI_SIMD_CALL uni_simd_execute(uni_simd_kernel_e kernel,
                                                              const void* input,
                                                              void* output,
                                                              uni_simd_param_t* params,
                                                              size_t params_len,
                                                              uni_simd_state_t** state);

#ifdef __cplusplus
}
#endif
