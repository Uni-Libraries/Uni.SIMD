#include "ifft_cf32/ifft_cf32_internal.hpp"

#include <cstddef>

#include <immintrin.h>

namespace uni::simd::detail {
namespace {

constexpr float root_half = 0.70710678118654752440f;

struct Complex4 final {
    float real[4U];
    float imag[4U];
};

[[nodiscard]] inline Complex4 ifft4_stride(const float* const real, const float* const imag,
                                           const std::size_t offset) noexcept {
    const float sum02_re = real[offset] + real[offset + 4U];
    const float sum02_im = imag[offset] + imag[offset + 4U];
    const float diff02_re = real[offset] - real[offset + 4U];
    const float diff02_im = imag[offset] - imag[offset + 4U];
    const float sum13_re = real[offset + 2U] + real[offset + 6U];
    const float sum13_im = imag[offset + 2U] + imag[offset + 6U];
    const float diff13_re = real[offset + 2U] - real[offset + 6U];
    const float diff13_im = imag[offset + 2U] - imag[offset + 6U];
    return {
        .real = {sum02_re + sum13_re, diff02_re - diff13_im,
                 sum02_re - sum13_re, diff02_re + diff13_im},
        .imag = {sum02_im + sum13_im, diff02_im + diff13_re,
                 sum02_im - sum13_im, diff02_im - diff13_re},
    };
}

} // namespace

void Ifft8_avx2_fma(float* const real, float* const imag, const std::size_t count) noexcept {
    if (count != 8U) {
        Ifft_generic(real, imag, count);
        return;
    }

    const Complex4 even = ifft4_stride(real, imag, 0U);
    const Complex4 odd = ifft4_stride(real, imag, 1U);
    const __m128 even_re = _mm_loadu_ps(even.real);
    const __m128 even_im = _mm_loadu_ps(even.imag);
    const __m128 odd_re = _mm_loadu_ps(odd.real);
    const __m128 odd_im = _mm_loadu_ps(odd.imag);
    const __m128 rotation_re = _mm_setr_ps(1.0f, root_half, 0.0f, -root_half);
    const __m128 rotation_im = _mm_setr_ps(0.0f, root_half, 1.0f, root_half);
    const __m128 product_re = _mm_fmsub_ps(odd_re, rotation_re, _mm_mul_ps(odd_im, rotation_im));
    const __m128 product_im = _mm_fmadd_ps(odd_re, rotation_im, _mm_mul_ps(odd_im, rotation_re));
    _mm256_storeu_ps(real, _mm256_set_m128(_mm_sub_ps(even_re, product_re),
                                           _mm_add_ps(even_re, product_re)));
    _mm256_storeu_ps(imag, _mm256_set_m128(_mm_sub_ps(even_im, product_im),
                                           _mm_add_ps(even_im, product_im)));
}

} // namespace uni::simd::detail
