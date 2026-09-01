#include "qpsk_carrier_analyzer_internal.hpp"

#include <cmath>

namespace uni::simd::kernels {

void QpskCarrierAnalyzer_scalar_accumulate(uni_simd_qpsk_carrier_analyzer_t& analyzer, const float* const input, const std::size_t count,
                                           uni_simd_qpsk_carrier_analyzer_result_t& result) noexcept {
    for (std::size_t index = 0U; index < count; ++index) {
        const float real = input[2U * index];
        const float imag = input[2U * index + 1U];
        const float power = std::fma(real, real, imag * imag);
        result.input_power += power;

        const float decision_real = std::copysign(1.0f, real);
        const float decision_imag = std::copysign(1.0f, imag);
        result.decision_sum[0] = std::fma(real, decision_real, std::fma(imag, decision_imag, result.decision_sum[0]));
        result.decision_sum[1] = std::fma(imag, decision_real, std::fma(-real, decision_imag, result.decision_sum[1]));

        if (!(power > analyzer.magnitude_epsilon)) {
            analyzer.previous_fourth_valid = false;
            continue;
        }

        const float square_real = std::fma(real, real, -(imag * imag)) / power;
        const float square_imag = (2.0f * real * imag) / power;
        const float fourth_real = std::fma(square_real, square_real, -(square_imag * square_imag));
        const float fourth_imag = 2.0f * square_real * square_imag;
        result.fourth_sum[0] += fourth_real;
        result.fourth_sum[1] += fourth_imag;
        result.valid_fourth_weight += power;
        ++result.valid_fourth_count;

        if (analyzer.previous_fourth_valid) {
            result.adjacent_fourth_sum[0] =
                std::fma(fourth_real, analyzer.previous_fourth_real, std::fma(fourth_imag, analyzer.previous_fourth_imag, result.adjacent_fourth_sum[0]));
            result.adjacent_fourth_sum[1] =
                std::fma(fourth_imag, analyzer.previous_fourth_real, std::fma(-fourth_real, analyzer.previous_fourth_imag, result.adjacent_fourth_sum[1]));
            ++result.adjacent_fourth_count;
        }
        analyzer.previous_fourth_real = fourth_real;
        analyzer.previous_fourth_imag = fourth_imag;
        analyzer.previous_fourth_valid = true;
    }
}

void QpskCarrierAnalyzer_generic(uni_simd_qpsk_carrier_analyzer_t& analyzer, const float* const input, const std::size_t count,
                                 uni_simd_qpsk_carrier_analyzer_result_t& result) noexcept {
    QpskCarrierAnalyzer_scalar_accumulate(analyzer, input, count, result);
}

} // namespace uni::simd::kernels
