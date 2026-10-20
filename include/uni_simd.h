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
 * Stateful-kernel creation parameters cannot be changed after the first
 * successful execution has created state.
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
 * Costas/analyzer/symbol synchronizer protocol:
 * - Set CONST_POINTER CONFIG to the matching configuration descriptor before
 *   first execution. The descriptor is copied when state is created.
 * - Costas input points to uni_simd_qpsk_costas4_block_t; output may be NULL or
 *   point to uni_simd_qpsk_costas4_state_t.
 * - Analyzer input points to uni_simd_qpsk_carrier_analyzer_block_t and output
 *   points to uni_simd_qpsk_carrier_analyzer_result_t.
 * - Symbol synchronizer input points to uni_simd_symbol_sync4_block_t and output
 *   points to uni_simd_symbol_sync4_result_t.
 * - Viterbi input points to uni_simd_viterbi_k7r2_block_t and output points to
 *   uni_simd_viterbi_k7r2_result_t; uni_simd_kernel_reset() restores the start metrics.
 * - uni_simd_kernel_reset() restores Costas or symbol synchronizer initial state
 *   (the synchronizer also forgets its sample history) or clears analyzer
 *   adjacency state.
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
 * Resets an initialized stateful kernel. Returns INVALID_ARGUMENT for stateless
 * kernels and INVALID_STATE before state has been created. For PFB, a pending
 * legacy RESET parameter is cleared; other one-shot parameters are unchanged.
 */
UNI_SIMD_API uni_simd_result_e UNI_SIMD_CALL uni_simd_kernel_reset(
    uni_simd_kernel_t* kernel);

/**
 * PFB channelizer: take over the stream position of `source`, an initialized
 * PFB kernel with the same configuration, so that both produce identical output
 * for identical input from now on. Returns INVALID_STATE before either kernel
 * has state and INVALID_ARGUMENT for other kernels or differing configurations.
 */
UNI_SIMD_API uni_simd_result_e UNI_SIMD_CALL uni_simd_pfb_channelizer_copy_state(uni_simd_kernel_t* kernel, const uni_simd_kernel_t* source);

/**
 * PFB channelizer: consume CF32 input (`input->count` complex samples, as for
 * execute) without producing the corresponding outputs. The kernel then continues the
 * stream exactly as if it had processed that input. With copy_state this lets
 * two kernels filter the two halves of one block concurrently.
 */
UNI_SIMD_API uni_simd_result_e UNI_SIMD_CALL uni_simd_pfb_channelizer_advance(uni_simd_kernel_t* kernel, const uni_simd_const_buffer_t* input);

/**
 * Stateful kernels: take over the complete streaming state of `source`, an initialized kernel of
 * the same kind and configuration, so that both continue identically from here. Supported for
 * PFB channelizer, four-channel QPSK Costas and four-channel symbol synchronizer kernels. Returns INVALID_STATE before
 * either kernel has state and INVALID_ARGUMENT for other kernels or differing configurations.
 */
UNI_SIMD_API uni_simd_result_e UNI_SIMD_CALL uni_simd_kernel_copy_state(uni_simd_kernel_t* kernel, const uni_simd_kernel_t* source);

/**
 * Four-channel symbol synchronizer: move past the input of `block` without producing symbols.
 * The timing loops coast at their current omega (the next symbol lands on the extrapolated
 * symbol grid, and the coasted symbols count as produced), the previous symbol is forgotten,
 * and the end of the input becomes the history
 * the next call interpolates from. Used to start a second instance part-way into a block, whose
 * loops then settle on the stream within a few loop time constants. Outputs are ignored.
 */
UNI_SIMD_API uni_simd_result_e UNI_SIMD_CALL uni_simd_symbol_sync4_skip(uni_simd_kernel_t* kernel, const uni_simd_symbol_sync4_block_t* block);

/**
 * Four-channel QPSK Costas: move the loops past `sample_count` samples without processing them.
 * The oscillator coasts: the steps already decided are applied, then every lane advances by its
 * current frequency per sample, as the loop would with a zero phase error. Used to start a
 * second instance part-way into a block, whose loops then settle on the carrier within a few
 * loop time constants (on any of the four QPSK phase positions, see rotate).
 */
UNI_SIMD_API uni_simd_result_e UNI_SIMD_CALL uni_simd_qpsk_costas4_skip(uni_simd_kernel_t* kernel, size_t sample_count);

/**
 * Four-channel QPSK Costas: advance the oscillator phase of lane k by quarter_turns[k] * pi / 2
 * (taken modulo 4). A Costas loop locks equally well on each of the four; this moves a loop to
 * another one without disturbing it, and multiplies all later outputs of the lane by
 * (-i)^quarter_turns[k]. Exact: the phasor components are swapped and negated.
 */
UNI_SIMD_API uni_simd_result_e UNI_SIMD_CALL uni_simd_qpsk_costas4_rotate(uni_simd_kernel_t* kernel, const int32_t* quarter_turns);

#ifdef __cplusplus
}
#endif
