#pragma once

#include "common/api_internal.hpp"
#include "uni_simd_typedefs.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

struct uni_simd_qpsk_carrier_analyzer_t;

namespace uni::simd::kernels {

using QpskCarrierAnalyze = void (*)(uni_simd_qpsk_carrier_analyzer_t&, const float*, std::size_t, uni_simd_qpsk_carrier_analyzer_result_t&) noexcept;

void QpskCarrierAnalyzer_generic(uni_simd_qpsk_carrier_analyzer_t& analyzer, const float* input, std::size_t count,
                                 uni_simd_qpsk_carrier_analyzer_result_t& result) noexcept;
void QpskCarrierAnalyzer_scalar_accumulate(uni_simd_qpsk_carrier_analyzer_t& analyzer, const float* input, std::size_t count,
                                            uni_simd_qpsk_carrier_analyzer_result_t& result) noexcept;

[[nodiscard]] uni_simd_result_e QpskCarrierAnalyzerInitialize(
    uni_simd_qpsk_carrier_analyzer_t& analyzer,
    const uni_simd_qpsk_carrier_analyzer_config_t& config,
    uni_simd_backend_e requested_backend,
    uni_simd_math_mode_e math_mode,
    bool prefer_energy_efficiency) noexcept;
[[nodiscard]] uni_simd_result_e QpskCarrierAnalyzerReset(
    uni_simd_qpsk_carrier_analyzer_t& analyzer) noexcept;
[[nodiscard]] uni_simd_result_e QpskCarrierAnalyzerExecute(
    uni_simd_qpsk_carrier_analyzer_t& analyzer,
    const uni_simd_qpsk_carrier_analyzer_block_t& block,
    uni_simd_qpsk_carrier_analyzer_result_t& result) noexcept;

#if UNI_SIMD_HAVE_AVX2_FMA
void QpskCarrierAnalyzer_avx2_fma(uni_simd_qpsk_carrier_analyzer_t& analyzer, const float* input, std::size_t count,
                                  uni_simd_qpsk_carrier_analyzer_result_t& result) noexcept;
#endif
#if UNI_SIMD_HAVE_AVX512F
void QpskCarrierAnalyzer_avx512f(uni_simd_qpsk_carrier_analyzer_t& analyzer, const float* input, std::size_t count,
                                 uni_simd_qpsk_carrier_analyzer_result_t& result) noexcept;
#endif
#if UNI_SIMD_HAVE_NEON
void QpskCarrierAnalyzer_neon(uni_simd_qpsk_carrier_analyzer_t& analyzer, const float* input, std::size_t count,
                              uni_simd_qpsk_carrier_analyzer_result_t& result) noexcept;
#endif

} // namespace uni::simd::kernels

struct uni_simd_qpsk_carrier_analyzer_t {
    float magnitude_epsilon{};
    float previous_fourth_real{};
    float previous_fourth_imag{};
    bool previous_fourth_valid{};
    uni::simd::kernels::QpskCarrierAnalyze analyze{};
    uni_simd_backend_e backend{UNI_SIMD_BACKEND_GENERIC};
};

namespace uni::simd::kernels {

template <std::size_t LaneCount>
inline void QpskCarrierAnalyzer_commit_adjacency(
    uni_simd_qpsk_carrier_analyzer_t& analyzer,
    uni_simd_qpsk_carrier_analyzer_result_t& result,
    const float (&fourth_real)[LaneCount],
    const float (&fourth_imag)[LaneCount],
    const std::uint32_t valid_mask) noexcept {
    for (std::size_t lane = 0U; lane < LaneCount; ++lane) {
        if ((valid_mask & (1U << lane)) == 0U) {
            analyzer.previous_fourth_valid = false;
            continue;
        }
        const float real = fourth_real[lane];
        const float imag = fourth_imag[lane];
        if (analyzer.previous_fourth_valid) {
            result.adjacent_fourth_sum[0] =
                std::fma(real, analyzer.previous_fourth_real,
                         std::fma(imag, analyzer.previous_fourth_imag, result.adjacent_fourth_sum[0]));
            result.adjacent_fourth_sum[1] =
                std::fma(imag, analyzer.previous_fourth_real,
                         std::fma(-real, analyzer.previous_fourth_imag, result.adjacent_fourth_sum[1]));
            ++result.adjacent_fourth_count;
        }
        analyzer.previous_fourth_real = real;
        analyzer.previous_fourth_imag = imag;
        analyzer.previous_fourth_valid = true;
    }
}

template <std::size_t LaneCount>
inline void QpskCarrierAnalyzer_commit_batch(uni_simd_qpsk_carrier_analyzer_t& analyzer, uni_simd_qpsk_carrier_analyzer_result_t& result,
                                             const float (&power)[LaneCount], const float (&decision_real)[LaneCount], const float (&decision_imag)[LaneCount],
                                             const float (&fourth_real)[LaneCount], const float (&fourth_imag)[LaneCount],
                                             const std::uint32_t valid_mask) noexcept {
    for (std::size_t lane = 0U; lane < LaneCount; ++lane) {
        result.input_power += power[lane];
        result.decision_sum[0] += decision_real[lane];
        result.decision_sum[1] += decision_imag[lane];
        if ((valid_mask & (1U << lane)) == 0U) {
            continue;
        }
        const float real = fourth_real[lane];
        const float imag = fourth_imag[lane];
        result.fourth_sum[0] += real;
        result.fourth_sum[1] += imag;
        result.valid_fourth_weight += power[lane];
        ++result.valid_fourth_count;
    }
    QpskCarrierAnalyzer_commit_adjacency(analyzer, result, fourth_real, fourth_imag, valid_mask);
}

} // namespace uni::simd::kernels
