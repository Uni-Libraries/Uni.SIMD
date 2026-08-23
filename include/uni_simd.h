#pragma once

#include "uni_simd_export.h"
#include "uni_simd_kernels.h"
#include "uni_simd_typedefs.h"
#include "uni_simd_version.h"

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
 * The function is thread-safe and idempotent. Kernel creation and operations
 * before successful initialization fail.
 */
UNI_SIMD_API uni_simd_result_e UNI_SIMD_CALL uni_simd_initialize(void);

/**
 * Finalizes the process-wide runtime.
 *
 * The function is thread-safe and idempotent. It returns
 * UNI_SIMD_RESULT_INVALID_STATE without finalizing while any kernel instance is
 * alive. Release every instance through uni_simd_kernel_free() first.
 */
UNI_SIMD_API uni_simd_result_e UNI_SIMD_CALL uni_simd_finalize(void);

/**
 * Creates a kernel instance.
 *
 * Returns NULL for an invalid ID, unavailable runtime, or allocation failure.
 */
UNI_SIMD_API uni_simd_kernel_t* UNI_SIMD_CALL uni_simd_kernel_create(uni_simd_kernel_e kernel);

/**
 * Sets or replaces one kernel parameter. The value field is inferred from the
 * parameter ID. Borrowed pointers must remain valid while the library may
 * access them. PFB configuration arrays are copied when streaming state is
 * created. The caller must serialize this call with every other operation on
 * the same kernel instance.
 */
UNI_SIMD_API uni_simd_result_e UNI_SIMD_CALL uni_simd_kernel_param_set(
    uni_simd_kernel_t* kernel, uni_simd_param_t param);

/** Atomically applies a parameter batch. Calls on one instance require caller serialization. */
UNI_SIMD_API uni_simd_result_e UNI_SIMD_CALL uni_simd_kernel_param_set_many(
    uni_simd_kernel_t* kernel, uni_simd_param_t* params, size_t param_count);

/**
 * Releases a kernel instance. Passing NULL is a successful no-op. The caller
 * must ensure no operation using this instance is running concurrently.
 */
UNI_SIMD_API uni_simd_result_e UNI_SIMD_CALL uni_simd_kernel_free(uni_simd_kernel_t* kernel);

/**
 * Executes a configured kernel instance.
 *
 * Common optional parameters are U32 BACKEND (default AUTOMATIC), U32
 * MATH_MODE (default FAST), U32 PREFER_ENERGY_EFFICIENCY (0 or 1), and POINTER
 * RESOLVED_BACKEND. The pointer receives the implementation that actually ran.
 * For PFB, creation parameters cannot be changed after its first successful
 * execution has created streaming state.
 *
 * Input/output descriptor types and kernel-specific parameters are documented
 * in uni_simd_kernels.h.
 *
 * PFB protocol:
 * - Configuration: required parameters are SIZE BIN_COUNT,
 *   SIZE DECIMATION, U32 GRID_OFFSET, CONST_POINTER TAPS, SIZE TAP_COUNT,
 *   CONST_POINTER LOGICAL_BINS, and SIZE LOGICAL_BIN_COUNT. Configuration is
 *   copied when streaming state is created. input/output may be NULL on the
 *   first execution to create state without processing.
 * - Processing: input points to uni_simd_const_buffer_t whose data is a flat
 *   interleaved {real, imaginary} float array; output points to
 *   uni_simd_buffer_array_t with one equally formatted buffer per logical bin.
 *   Buffer counts are complex-sample counts. POINTER OUTPUT_COUNT is an optional
 *   output parameter receiving samples produced per output.
 * - Query: U32 QUERY_OUTPUT_COUNT=1 and a POINTER OUTPUT_COUNT parameter compute
 *   the next output count from input->count without advancing state; output is
 *   ignored and input->data may be NULL. QUERY_OUTPUT_COUNT is consumed by the
 *   next execution attempt.
 * - Reset: U32 RESET=1 clears history immediately before processing, after all
 *   descriptors and capacities have been validated. RESET is consumed by the
 *   next execution attempt. Reset and query cannot be combined.
 * - Destruction: release the kernel with uni_simd_kernel_free().
 *
 * Buffers are borrowed for the duration of the call and require their element
 * type's natural alignment, but no additional SIMD alignment. Except for COPY_U8 and
 * documented exact in-place kernels, active input and output ranges must not
 * overlap. Operations on one kernel instance are not internally synchronized:
 * the caller must not invoke execute, parameter updates, reset, or free
 * concurrently for the same instance. Separate instances may execute concurrently.
 */
UNI_SIMD_API uni_simd_result_e UNI_SIMD_CALL uni_simd_kernel_execute(
    uni_simd_kernel_t* kernel, const void* input, void* output);

/**
 * Resets an initialized PFB kernel. Returns INVALID_ARGUMENT for other kernels
 * and INVALID_STATE before PFB streaming state has been created. A pending
 * legacy RESET parameter is cleared; other one-shot parameters are unchanged.
 */
UNI_SIMD_API uni_simd_result_e UNI_SIMD_CALL uni_simd_kernel_reset(
    uni_simd_kernel_t* kernel);

#ifdef __cplusplus
}
#endif
