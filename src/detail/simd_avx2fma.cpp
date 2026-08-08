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
    const __m128 t = _mm_loadu_ps(taps);
    const __m128 lo = _mm_unpacklo_ps(t, t);
    const __m128 hi = _mm_unpackhi_ps(t, t);
    __m256 out = _mm256_castps128_ps256(lo);
    out = _mm256_insertf128_ps(out, hi, 1);
    return out;
}

[[nodiscard]] inline std::complex<float> reduce_complex_acc_avx2(__m256 acc) noexcept {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    return {tmp[0] + tmp[2] + tmp[4] + tmp[6], tmp[1] + tmp[3] + tmp[5] + tmp[7]};
}
} // namespace

namespace uni::simd::detail {

std::complex<float> DotProdSymmetricCF32Real_avx2fma(const void* src, const float* taps_pairs, size_t pair_count, float center_tap) noexcept {
    std::complex<float> out{};
    if (!src) {
        return out;
    }

    const auto* x = static_cast<const float*>(src);
    const size_t center_off = 2U * pair_count;
    out.real(x[center_off + 0U] * center_tap);
    out.imag(x[center_off + 1U] * center_tap);

    if (pair_count == 0U) {
        return out;
    }

    const size_t tail_off = 4U * pair_count;
    const __m256i reverse_complex4 = _mm256_setr_epi32(6, 7, 4, 5, 2, 3, 0, 1);
    __m256 acc = _mm256_setzero_ps();
    size_t k = 0;

    for (; k + 4U <= pair_count; k += 4U) {
        const __m256 left = _mm256_loadu_ps(x + 2U * k);
        __m256 right = _mm256_loadu_ps(x + (tail_off - (2U * k + 6U)));
        right = _mm256_permutevar8x32_ps(right, reverse_complex4);
        const __m256 pair_sum = _mm256_add_ps(left, right);
        const __m256 tv = duplicate_real_taps4_avx2(taps_pairs + k);
        acc = _mm256_fmadd_ps(pair_sum, tv, acc);
    }

    const std::complex<float> vec_acc = reduce_complex_acc_avx2(acc);
    out.real(out.real() + vec_acc.real());
    out.imag(out.imag() + vec_acc.imag());

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
