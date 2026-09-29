#pragma once

#include "common/api_internal.hpp"
#include "uni_simd_typedefs.h"

#include <cstddef>
#include <cstdint>

struct uni_simd_symbol_sync4_t;

namespace uni::simd::kernels {

inline constexpr std::size_t kSymbolSync4Lanes = UNI_SIMD_SYMBOL_SYNC4_CHANNEL_COUNT;

/**
 * Samples kept from the previous block (and taken from the start of the current one to build
 * the seam, see SymbolSync4Execute). Enough for a Gardner mid-point half a symbol behind a symbol
 * that starts just before the block, plus the four-sample stencil: ceil(omega_max / 2) + 8,
 * rounded up to a multiple of four.
 */
inline constexpr std::size_t kSymbolSync4MaxHistory = 64U;

/**
 * Timing loop state, one entry per lane, laid out as structure of arrays so the vector backend
 * loads each field with one instruction. Counters are doubles for the same reason (exact far
 * beyond any realistic symbol count).
 */
struct SymbolSync4State final {
    /** Position of the next symbol in samples, relative to the start of the segment being processed. */
    alignas(32) double time[kSymbolSync4Lanes];
    alignas(32) double omega[kSymbolSync4Lanes];
    alignas(32) double err_norm2_ema[kSymbolSync4Lanes];
    /** Consecutive symbols below the lock threshold (Gardner lanes) / above the fallback threshold (M&M lanes). */
    alignas(32) double lock_run[kSymbolSync4Lanes];
    alignas(32) double unlock_run[kSymbolSync4Lanes];
    alignas(32) double symbols[kSymbolSync4Lanes];
    /** Last symbol per lane, interleaved {re, im}. */
    alignas(32) float previous[2U * kSymbolSync4Lanes];
    std::uint32_t have_previous[kSymbolSync4Lanes];
    /** Active detector: 0 Gardner, 1 Mueller and Muller. */
    std::uint32_t mueller_muller[kSymbolSync4Lanes];
    std::uint64_t ted_switches[kSymbolSync4Lanes];
};

/** Loop constants derived from the configuration. */
struct SymbolSync4Params final {
    alignas(32) double omega_min[kSymbolSync4Lanes];
    alignas(32) double omega_max[kSymbolSync4Lanes];
    alignas(32) double alpha[kSymbolSync4Lanes];
    alignas(32) double beta[kSymbolSync4Lanes];
    /**
     * alpha + beta. While omega stays inside its bounds, t_{k+1} = (t_k + omega_{k-1}) +
     * (alpha + beta) * e_k, which puts one multiply-add instead of three operations between the
     * detector error and the next symbol position.
     */
    alignas(32) double alpha_beta[kSymbolSync4Lanes];
    bool cubic{true};
    bool auto_ted{};
    bool auto_ted_fallback{};
    double lock_ema_alpha{};
    double mm_min_symbols{};
    /** Hold counts as doubles; a disabled switch (hold 0) becomes +infinity so it never fires. */
    double mm_hold_symbols{};
    double fallback_hold_symbols{};
    double mm_threshold{};
    double fallback_threshold{};
};

/**
 * One pass of the timing loops over a contiguous sample range per lane. A lane produces symbols
 * while `time < limit` and its output has room. Its interpolation stencils must lie inside
 * `data[0, length)`; backends clamp the stencil into that range so a corrupted time can never
 * read outside it.
 */
struct SymbolSync4Segments final {
    const float* data[kSymbolSync4Lanes];
    std::size_t length[kSymbolSync4Lanes];
    double limit[kSymbolSync4Lanes];
    float* output[kSymbolSync4Lanes];
    std::size_t output_capacity[kSymbolSync4Lanes];
    /** Symbols already in `output`; advanced by the pass. */
    std::size_t output_count[kSymbolSync4Lanes];
};

using SymbolSync4Run = void (*)(const SymbolSync4Params&, SymbolSync4State&, SymbolSync4Segments&) noexcept;

void SymbolSync4_generic(const SymbolSync4Params& params, SymbolSync4State& state, SymbolSync4Segments& segments) noexcept;
#if UNI_SIMD_HAVE_AVX2_FMA
void SymbolSync4_avx2(const SymbolSync4Params& params, SymbolSync4State& state, SymbolSync4Segments& segments) noexcept;
#endif

/** Detector switch of one lane (shared by the backends: switches are rare). */
void SymbolSync4SwitchTed(SymbolSync4State& state, std::size_t lane) noexcept;

[[nodiscard]] uni_simd_result_e SymbolSync4Initialize(uni_simd_symbol_sync4_t& kernel, const uni_simd_symbol_sync4_config_t& config,
                                                      uni_simd_backend_e requested_backend, uni_simd_math_mode_e math_mode) noexcept;
[[nodiscard]] uni_simd_result_e SymbolSync4Reset(uni_simd_symbol_sync4_t& kernel) noexcept;
[[nodiscard]] uni_simd_result_e SymbolSync4Execute(uni_simd_symbol_sync4_t& kernel, const uni_simd_symbol_sync4_block_t& block,
                                                   uni_simd_symbol_sync4_result_t& result) noexcept;

} // namespace uni::simd::kernels

struct uni_simd_symbol_sync4_t {
    uni_simd_symbol_sync4_config_t config{};
    uni::simd::kernels::SymbolSync4Params params{};
    uni::simd::kernels::SymbolSync4State initial_state{};
    uni::simd::kernels::SymbolSync4State state{};
    uni::simd::kernels::SymbolSync4Run run{};
    uni_simd_backend_e backend{UNI_SIMD_BACKEND_GENERIC};
    std::size_t history_length{};
    /** Last `history_length` samples of every lane, interleaved CF32. */
    float history[uni::simd::kernels::kSymbolSync4Lanes][2U * uni::simd::kernels::kSymbolSync4MaxHistory]{};
    /** History followed by the first `history_length` samples of the current block. */
    float seam[uni::simd::kernels::kSymbolSync4Lanes][4U * uni::simd::kernels::kSymbolSync4MaxHistory]{};
};
