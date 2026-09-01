#include "qpsk_carrier_analyzer_internal.hpp"

#if UNI_SIMD_HAVE_AVX2_FMA

#include <immintrin.h>

#include <bit>

namespace uni::simd::kernels {
namespace {

struct Components8 final {
    __m256 real;
    __m256 imag;
};

[[nodiscard]] inline Components8 Load8(const float* const input) noexcept {
    const __m256 first = _mm256_loadu_ps(input);
    const __m256 second = _mm256_loadu_ps(input + 8U);
    const __m128 first_low = _mm256_castps256_ps128(first);
    const __m128 first_high = _mm256_extractf128_ps(first, 1);
    const __m128 second_low = _mm256_castps256_ps128(second);
    const __m128 second_high = _mm256_extractf128_ps(second, 1);
    __m256 real = _mm256_castps128_ps256(_mm_shuffle_ps(first_low, first_high, _MM_SHUFFLE(2, 0, 2, 0)));
    real = _mm256_insertf128_ps(real, _mm_shuffle_ps(second_low, second_high, _MM_SHUFFLE(2, 0, 2, 0)), 1);
    __m256 imag = _mm256_castps128_ps256(_mm_shuffle_ps(first_low, first_high, _MM_SHUFFLE(3, 1, 3, 1)));
    imag = _mm256_insertf128_ps(imag, _mm_shuffle_ps(second_low, second_high, _MM_SHUFFLE(3, 1, 3, 1)), 1);
    return {real, imag};
}

} // namespace

void QpskCarrierAnalyzer_avx2_fma(uni_simd_qpsk_carrier_analyzer_t& analyzer, const float* input, const std::size_t count,
                                  uni_simd_qpsk_carrier_analyzer_result_t& result) noexcept {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256 epsilon = _mm256_set1_ps(analyzer.magnitude_epsilon);
    __m256 power_sum = zero;
    __m256 valid_weight_sum = zero;
    __m256 decision_real_sum = zero;
    __m256 decision_imag_sum = zero;
    __m256 fourth_real_sum = zero;
    __m256 fourth_imag_sum = zero;

    std::size_t index = 0U;
    for (; index + 8U <= count; index += 8U) {
        const Components8 components = Load8(input + 2U * index);
        const __m256 real = components.real;
        const __m256 imag = components.imag;
        const __m256 power = _mm256_fmadd_ps(real, real, _mm256_mul_ps(imag, imag));
        const __m256 decision_real_sign = _mm256_or_ps(one, _mm256_and_ps(real, sign));
        const __m256 decision_imag_sign = _mm256_or_ps(one, _mm256_and_ps(imag, sign));
        const __m256 decision_real = _mm256_fmadd_ps(real, decision_real_sign, _mm256_mul_ps(imag, decision_imag_sign));
        const __m256 decision_imag = _mm256_fmsub_ps(imag, decision_real_sign, _mm256_mul_ps(real, decision_imag_sign));

        const __m256 valid = _mm256_cmp_ps(power, epsilon, _CMP_GT_OQ);
        const __m256 safe_power = _mm256_blendv_ps(one, power, valid);
        const __m256 square_real = _mm256_div_ps(_mm256_fmsub_ps(real, real, _mm256_mul_ps(imag, imag)), safe_power);
        const __m256 square_imag = _mm256_div_ps(_mm256_mul_ps(two, _mm256_mul_ps(real, imag)), safe_power);
        __m256 fourth_real = _mm256_fmsub_ps(square_real, square_real, _mm256_mul_ps(square_imag, square_imag));
        __m256 fourth_imag = _mm256_mul_ps(two, _mm256_mul_ps(square_real, square_imag));
        fourth_real = _mm256_blendv_ps(zero, fourth_real, valid);
        fourth_imag = _mm256_blendv_ps(zero, fourth_imag, valid);
        power_sum = _mm256_add_ps(power_sum, power);
        valid_weight_sum = _mm256_add_ps(valid_weight_sum, _mm256_blendv_ps(zero, power, valid));
        decision_real_sum = _mm256_add_ps(decision_real_sum, decision_real);
        decision_imag_sum = _mm256_add_ps(decision_imag_sum, decision_imag);
        fourth_real_sum = _mm256_add_ps(fourth_real_sum, fourth_real);
        fourth_imag_sum = _mm256_add_ps(fourth_imag_sum, fourth_imag);

        alignas(32) float fourths_real[8];
        alignas(32) float fourths_imag[8];
        _mm256_store_ps(fourths_real, fourth_real);
        _mm256_store_ps(fourths_imag, fourth_imag);
        const auto valid_mask = static_cast<std::uint32_t>(_mm256_movemask_ps(valid));
        result.valid_fourth_count += std::popcount(valid_mask);
        QpskCarrierAnalyzer_commit_adjacency(analyzer, result, fourths_real, fourths_imag, valid_mask);
    }
    alignas(32) float sums[6][8];
    _mm256_store_ps(sums[0], power_sum);
    _mm256_store_ps(sums[1], valid_weight_sum);
    _mm256_store_ps(sums[2], decision_real_sum);
    _mm256_store_ps(sums[3], decision_imag_sum);
    _mm256_store_ps(sums[4], fourth_real_sum);
    _mm256_store_ps(sums[5], fourth_imag_sum);
    for (std::size_t lane = 0U; lane < 8U; ++lane) {
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
