//
// Includes
//

// stdlib
#include <cstring>

// compiler
#include <immintrin.h>

#include "detail/simd_avx2.h"
#include "detail/simd_avx2fma.h"
#include "detail/simd_generic.h"
#include "detail/simd_sse2.h"



//
// Functions
//

namespace {
[[nodiscard]] inline __m256 duplicate_real_taps4_avx2(const float* taps) noexcept {
    // Match four real taps to the interleaved {real, imag} components of four complex samples.
    const __m128 t = _mm_loadu_ps(taps);
    const __m128 lo = _mm_unpacklo_ps(t, t);
    const __m128 hi = _mm_unpackhi_ps(t, t);
    __m256 out = _mm256_castps128_ps256(lo);
    out = _mm256_insertf128_ps(out, hi, 1);
    return out;
}

} // namespace

namespace uni::simd::detail {

std::complex<float> DotProdCF32Real_avx2fma(const void* src, const float* taps, size_t len) noexcept {
    if (!src || !taps || len == 0U) {
        return {};
    }

    const auto* x = static_cast<const float*>(src);
    const __m256i taps0_indices = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
    const __m256i taps1_indices = _mm256_setr_epi32(4, 4, 5, 5, 6, 6, 7, 7);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    size_t i = 0;

    // Load taps eight at a time and use four independent FMA chains to hide instruction latency.
    for (; i + 16U <= len; i += 16U) {
        const __m256 tap_values0 = _mm256_loadu_ps(taps + i);
        const __m256 tap_values1 = _mm256_loadu_ps(taps + i + 8U);
        const __m256 tv0 = _mm256_permutevar8x32_ps(tap_values0, taps0_indices);
        const __m256 tv1 = _mm256_permutevar8x32_ps(tap_values0, taps1_indices);
        const __m256 tv2 = _mm256_permutevar8x32_ps(tap_values1, taps0_indices);
        const __m256 tv3 = _mm256_permutevar8x32_ps(tap_values1, taps1_indices);
        const __m256 xv0 = _mm256_loadu_ps(x + 2U * (i + 0U));
        const __m256 xv1 = _mm256_loadu_ps(x + 2U * (i + 4U));
        const __m256 xv2 = _mm256_loadu_ps(x + 2U * (i + 8U));
        const __m256 xv3 = _mm256_loadu_ps(x + 2U * (i + 12U));
        acc0 = _mm256_fmadd_ps(xv0, tv0, acc0);
        acc1 = _mm256_fmadd_ps(xv1, tv1, acc1);
        acc2 = _mm256_fmadd_ps(xv2, tv2, acc2);
        acc3 = _mm256_fmadd_ps(xv3, tv3, acc3);
    }

    for (; i + 8U <= len; i += 8U) {
        const __m256 tap_values = _mm256_loadu_ps(taps + i);
        const __m256 tv0 = _mm256_permutevar8x32_ps(tap_values, taps0_indices);
        const __m256 tv1 = _mm256_permutevar8x32_ps(tap_values, taps1_indices);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + 2U * i), tv0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + 2U * i + 8U), tv1, acc1);
    }

    // Consume a final complete four-sample vector before the scalar tail.
    for (; i + 4U <= len; i += 4U) {
        const __m256 xv = _mm256_loadu_ps(x + 2U * i);
        acc0 = _mm256_fmadd_ps(xv, duplicate_real_taps4_avx2(taps + i), acc0);
    }

    const __m256 vec_acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    const __m128 lane_sum =
        _mm_add_ps(_mm256_castps256_ps128(vec_acc), _mm256_extractf128_ps(vec_acc, 1));
    const __m128 complex_sum = _mm_add_ps(lane_sum, _mm_movehl_ps(lane_sum, lane_sum));
    float acc_re = _mm_cvtss_f32(complex_sum);
    float acc_im = _mm_cvtss_f32(_mm_shuffle_ps(complex_sum, complex_sum, _MM_SHUFFLE(1, 1, 1, 1)));

    for (; i < len; ++i) {
        const float tap = taps[i];
        acc_re += x[2U * i + 0U] * tap;
        acc_im += x[2U * i + 1U] * tap;
    }

    return {acc_re, acc_im};
}

std::complex<float> DotProdSymmetricCF32Real_avx2fma(const void* src, const float* taps_pairs, size_t pair_count, float center_tap) noexcept {
    std::complex<float> out{};
    if (!src) {
        return out;
    }

    const auto* x = static_cast<const float*>(src);
    const size_t center_off = 2U * pair_count;

    // Seed the result with the center sample, which has no mirrored partner.
    out.real(x[center_off + 0U] * center_tap);
    out.imag(x[center_off + 1U] * center_tap);

    if (pair_count == 0U) {
        return out;
    }

    const size_t tail_off = 4U * pair_count;
    const __m256i reverse_complex4 = _mm256_setr_epi32(6, 7, 4, 5, 2, 3, 0, 1);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t k = 0;

    // Accumulate mirrored pairs through two independent FMA chains to hide instruction latency.
    for (; k + 8U <= pair_count; k += 8U) {
        const __m256 left0 = _mm256_loadu_ps(x + 2U * k);
        __m256 right0 = _mm256_loadu_ps(x + (tail_off - (2U * k + 6U)));
        right0 = _mm256_permutevar8x32_ps(right0, reverse_complex4);
        const __m256 pair_sum0 = _mm256_add_ps(left0, right0);
        const __m256 tv0 = duplicate_real_taps4_avx2(taps_pairs + k);
        acc0 = _mm256_fmadd_ps(pair_sum0, tv0, acc0);

        const __m256 left1 = _mm256_loadu_ps(x + 2U * (k + 4U));
        __m256 right1 = _mm256_loadu_ps(x + (tail_off - (2U * (k + 4U) + 6U)));
        right1 = _mm256_permutevar8x32_ps(right1, reverse_complex4);
        const __m256 pair_sum1 = _mm256_add_ps(left1, right1);
        const __m256 tv1 = duplicate_real_taps4_avx2(taps_pairs + k + 4U);
        acc1 = _mm256_fmadd_ps(pair_sum1, tv1, acc1);
    }

    // Consume a final complete four-pair vector before the scalar tail.
    for (; k + 4U <= pair_count; k += 4U) {
        const __m256 left = _mm256_loadu_ps(x + 2U * k);
        __m256 right = _mm256_loadu_ps(x + (tail_off - (2U * k + 6U)));
        right = _mm256_permutevar8x32_ps(right, reverse_complex4);
        const __m256 pair_sum = _mm256_add_ps(left, right);
        const __m256 tv = duplicate_real_taps4_avx2(taps_pairs + k);
        acc0 = _mm256_fmadd_ps(pair_sum, tv, acc0);
    }

    // Fold the interleaved SIMD lanes into the seeded center result.
    const __m256 vec_acc = _mm256_add_ps(acc0, acc1);
    const __m128 lane_sum =
        _mm_add_ps(_mm256_castps256_ps128(vec_acc), _mm256_extractf128_ps(vec_acc, 1));
    const __m128 complex_sum = _mm_add_ps(lane_sum, _mm_movehl_ps(lane_sum, lane_sum));
    out.real(out.real() + _mm_cvtss_f32(complex_sum));
    out.imag(out.imag() +
             _mm_cvtss_f32(_mm_shuffle_ps(complex_sum, complex_sum, _MM_SHUFFLE(1, 1, 1, 1))));

    // Finish mirrored pairs that do not fill a complete SIMD vector.
    for (; k < pair_count; ++k) {
        const float tap = taps_pairs[k];
        const size_t lo = 2U * k;
        const size_t hi = 4U * pair_count - 2U * k;
        out.real(out.real() + tap * (x[lo + 0U] + x[hi + 0U]));
        out.imag(out.imag() + tap * (x[lo + 1U] + x[hi + 1U]));
    }

    return out;
}

} // namespace uni::simd::detail
