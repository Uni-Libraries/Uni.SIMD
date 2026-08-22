#include "ifft_cf32/ifft_cf32_internal.hpp"

#include <cstddef>

#include <immintrin.h>

namespace uni::simd::detail {
namespace {

constexpr float root_half = 0.70710678118654752440f;

alignas(32) constexpr float cosine[8U][8U]{
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
    {1.0f, root_half, 0.0f, -root_half, -1.0f, -root_half, 0.0f, root_half},
    {1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f},
    {1.0f, -root_half, 0.0f, root_half, -1.0f, root_half, 0.0f, -root_half},
    {1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f},
    {1.0f, -root_half, 0.0f, root_half, -1.0f, root_half, 0.0f, -root_half},
    {1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f},
    {1.0f, root_half, 0.0f, -root_half, -1.0f, -root_half, 0.0f, root_half},
};

alignas(32) constexpr float sine[8U][8U]{
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, root_half, 1.0f, root_half, 0.0f, -root_half, -1.0f, -root_half},
    {0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, -1.0f},
    {0.0f, root_half, -1.0f, root_half, 0.0f, -root_half, 1.0f, -root_half},
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, -root_half, 1.0f, -root_half, 0.0f, root_half, -1.0f, root_half},
    {0.0f, -1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f},
    {0.0f, -root_half, -1.0f, -root_half, 0.0f, root_half, 1.0f, root_half},
};

} // namespace

void Ifft8_avx2_fma(float* const real, float* const imag, const std::size_t count) noexcept {
    if (count != 8U) {
        Ifft_generic(real, imag, count);
        return;
    }

    __m256 output_re = _mm256_setzero_ps();
    __m256 output_im = _mm256_setzero_ps();
    for (std::size_t input = 0U; input < 8U; ++input) {
        const __m256 twiddle_re = _mm256_load_ps(cosine[input]);
        const __m256 twiddle_im = _mm256_load_ps(sine[input]);
        const __m256 input_re = _mm256_broadcast_ss(real + input);
        const __m256 input_im = _mm256_broadcast_ss(imag + input);
        output_re = _mm256_fmadd_ps(input_re, twiddle_re, output_re);
        output_re = _mm256_fnmadd_ps(input_im, twiddle_im, output_re);
        output_im = _mm256_fmadd_ps(input_re, twiddle_im, output_im);
        output_im = _mm256_fmadd_ps(input_im, twiddle_re, output_im);
    }
    _mm256_storeu_ps(real, output_re);
    _mm256_storeu_ps(imag, output_im);
}

} // namespace uni::simd::detail
