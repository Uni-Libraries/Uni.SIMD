#include "qpsk_carrier_analyzer_internal.hpp"

#if UNI_SIMD_HAVE_AVX512F

#include <immintrin.h>

#include <bit>

namespace uni::simd::kernels {

void QpskCarrierAnalyzer_avx512f(uni_simd_qpsk_carrier_analyzer_t& analyzer, const float* input, const std::size_t count,
                                 uni_simd_qpsk_carrier_analyzer_result_t& result) noexcept {
    const __m512 zero = _mm512_setzero_ps();
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 two = _mm512_set1_ps(2.0f);
    const __m512i one_bits = _mm512_castps_si512(one);
    const __m512i sign_bits = _mm512_set1_epi32(-2147483647 - 1);
    const __m512 epsilon = _mm512_set1_ps(analyzer.magnitude_epsilon);
    const __m512i real_indices = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i imag_indices = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    __m512 power_sum = zero;
    __m512 valid_weight_sum = zero;
    __m512 decision_real_sum = zero;
    __m512 decision_imag_sum = zero;
    __m512 fourth_real_sum = zero;
    __m512 fourth_imag_sum = zero;

    std::size_t index = 0U;
    for (; index + 16U <= count; index += 16U) {
        const __m512 first = _mm512_loadu_ps(input + 2U * index);
        const __m512 second = _mm512_loadu_ps(input + 2U * index + 16U);
        const __m512 real = _mm512_permutex2var_ps(first, real_indices, second);
        const __m512 imag = _mm512_permutex2var_ps(first, imag_indices, second);
        const __m512 power = _mm512_fmadd_ps(real, real, _mm512_mul_ps(imag, imag));
        const __m512 decision_real_sign = _mm512_castsi512_ps(_mm512_or_si512(one_bits, _mm512_and_si512(_mm512_castps_si512(real), sign_bits)));
        const __m512 decision_imag_sign = _mm512_castsi512_ps(_mm512_or_si512(one_bits, _mm512_and_si512(_mm512_castps_si512(imag), sign_bits)));
        const __m512 decision_real = _mm512_fmadd_ps(real, decision_real_sign, _mm512_mul_ps(imag, decision_imag_sign));
        const __m512 decision_imag = _mm512_fmsub_ps(imag, decision_real_sign, _mm512_mul_ps(real, decision_imag_sign));

        const __mmask16 valid = _mm512_cmp_ps_mask(power, epsilon, _CMP_GT_OQ);
        const __m512 safe_power = _mm512_mask_blend_ps(valid, one, power);
        const __m512 square_real = _mm512_div_ps(_mm512_fmsub_ps(real, real, _mm512_mul_ps(imag, imag)), safe_power);
        const __m512 square_imag = _mm512_div_ps(_mm512_mul_ps(two, _mm512_mul_ps(real, imag)), safe_power);
        const __m512 fourth_real = _mm512_maskz_mov_ps(valid, _mm512_fmsub_ps(square_real, square_real, _mm512_mul_ps(square_imag, square_imag)));
        const __m512 fourth_imag = _mm512_maskz_mov_ps(valid, _mm512_mul_ps(two, _mm512_mul_ps(square_real, square_imag)));
        power_sum = _mm512_add_ps(power_sum, power);
        valid_weight_sum = _mm512_add_ps(valid_weight_sum, _mm512_maskz_mov_ps(valid, power));
        decision_real_sum = _mm512_add_ps(decision_real_sum, decision_real);
        decision_imag_sum = _mm512_add_ps(decision_imag_sum, decision_imag);
        fourth_real_sum = _mm512_add_ps(fourth_real_sum, fourth_real);
        fourth_imag_sum = _mm512_add_ps(fourth_imag_sum, fourth_imag);

        alignas(64) float fourths_real[16];
        alignas(64) float fourths_imag[16];
        _mm512_store_ps(fourths_real, fourth_real);
        _mm512_store_ps(fourths_imag, fourth_imag);
        result.valid_fourth_count += std::popcount(static_cast<std::uint32_t>(valid));
        QpskCarrierAnalyzer_commit_adjacency(analyzer, result, fourths_real, fourths_imag,
                                             static_cast<std::uint32_t>(valid));
    }
    alignas(64) float sums[6][16];
    _mm512_store_ps(sums[0], power_sum);
    _mm512_store_ps(sums[1], valid_weight_sum);
    _mm512_store_ps(sums[2], decision_real_sum);
    _mm512_store_ps(sums[3], decision_imag_sum);
    _mm512_store_ps(sums[4], fourth_real_sum);
    _mm512_store_ps(sums[5], fourth_imag_sum);
    for (std::size_t lane = 0U; lane < 16U; ++lane) {
        result.input_power += sums[0][lane];
        result.valid_fourth_weight += sums[1][lane];
        result.decision_sum[0] += sums[2][lane];
        result.decision_sum[1] += sums[3][lane];
        result.fourth_sum[0] += sums[4][lane];
        result.fourth_sum[1] += sums[5][lane];
    }
    QpskCarrierAnalyzer_scalar_accumulate(analyzer, input + 2U * index, count - index, result);
}

} // namespace uni::simd::kernels

#endif
