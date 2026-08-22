#include "ifft_cf32/ifft_cf32_internal.hpp"
#include "ifft_cf32/ifft_cf32_tables.hpp"

#include <cstddef>

#include <immintrin.h>

namespace uni::simd::detail {
namespace {

[[nodiscard]] inline __m256 stage2(const __m256 value) noexcept {
    const __m256 sign = _mm256_setr_ps(1.0f, -1.0f, 1.0f, -1.0f,
                                      1.0f, -1.0f, 1.0f, -1.0f);
    return _mm256_add_ps(_mm256_permute_ps(value, 0xB1), _mm256_mul_ps(sign, value));
}

inline void stage4(__m256& real, __m256& imag) noexcept {
    const __m256i even_indices = _mm256_setr_epi32(0, 1, 0, 1, 4, 5, 4, 5);
    const __m256i odd_indices = _mm256_setr_epi32(2, 3, 2, 3, 6, 7, 6, 7);
    const __m256 even_re = _mm256_permutevar8x32_ps(real, even_indices);
    const __m256 even_im = _mm256_permutevar8x32_ps(imag, even_indices);
    const __m256 odd_re = _mm256_permutevar8x32_ps(real, odd_indices);
    const __m256 odd_im = _mm256_permutevar8x32_ps(imag, odd_indices);
    const __m256 rotation_re = _mm256_setr_ps(1.0f, 0.0f, 1.0f, 0.0f,
                                              1.0f, 0.0f, 1.0f, 0.0f);
    const __m256 rotation_im = _mm256_setr_ps(0.0f, 1.0f, 0.0f, 1.0f,
                                              0.0f, 1.0f, 0.0f, 1.0f);
    const __m256 sign = _mm256_setr_ps(1.0f, 1.0f, -1.0f, -1.0f,
                                       1.0f, 1.0f, -1.0f, -1.0f);
    const __m256 product_re =
        _mm256_fmsub_ps(odd_re, rotation_re, _mm256_mul_ps(odd_im, rotation_im));
    const __m256 product_im =
        _mm256_fmadd_ps(odd_re, rotation_im, _mm256_mul_ps(odd_im, rotation_re));
    real = _mm256_fmadd_ps(sign, product_re, even_re);
    imag = _mm256_fmadd_ps(sign, product_im, even_im);
}

inline void stage8(__m256& real, __m256& imag) noexcept {
    constexpr float root_half = 0.70710678118654752440f;
    const __m256i even_indices = _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3);
    const __m256i odd_indices = _mm256_setr_epi32(4, 5, 6, 7, 4, 5, 6, 7);
    const __m256 even_re = _mm256_permutevar8x32_ps(real, even_indices);
    const __m256 even_im = _mm256_permutevar8x32_ps(imag, even_indices);
    const __m256 odd_re = _mm256_permutevar8x32_ps(real, odd_indices);
    const __m256 odd_im = _mm256_permutevar8x32_ps(imag, odd_indices);
    const __m256 rotation_re = _mm256_setr_ps(1.0f, root_half, 0.0f, -root_half,
                                              1.0f, root_half, 0.0f, -root_half);
    const __m256 rotation_im = _mm256_setr_ps(0.0f, root_half, 1.0f, root_half,
                                              0.0f, root_half, 1.0f, root_half);
    const __m256 sign = _mm256_setr_ps(1.0f, 1.0f, 1.0f, 1.0f,
                                       -1.0f, -1.0f, -1.0f, -1.0f);
    const __m256 product_re =
        _mm256_fmsub_ps(odd_re, rotation_re, _mm256_mul_ps(odd_im, rotation_im));
    const __m256 product_im =
        _mm256_fmadd_ps(odd_re, rotation_im, _mm256_mul_ps(odd_im, rotation_re));
    real = _mm256_fmadd_ps(sign, product_re, even_re);
    imag = _mm256_fmadd_ps(sign, product_im, even_im);
}

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

inline void ifft8(float* const real, float* const imag) noexcept {
    constexpr float root_half = 0.70710678118654752440f;
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

template <std::size_t Count>
inline void apply_late_stage(float* const real, float* const imag) noexcept {
    constexpr std::size_t half = Count / 2U;
    for (std::size_t base = 0U; base < Count; base += Count) {
        for (std::size_t index = 0U; index < half; index += 8U) {
            const __m256 even_re = _mm256_loadu_ps(real + base + index);
            const __m256 even_im = _mm256_loadu_ps(imag + base + index);
            const __m256 odd_re = _mm256_loadu_ps(real + base + half + index);
            const __m256 odd_im = _mm256_loadu_ps(imag + base + half + index);
            const __m256 rotation_re = _mm256_loadu_ps(ifft_stage_twiddle_re<Count>.data() + index);
            const __m256 rotation_im = _mm256_loadu_ps(ifft_stage_twiddle_im<Count>.data() + index);
            const __m256 product_re =
                _mm256_fmsub_ps(odd_re, rotation_re, _mm256_mul_ps(odd_im, rotation_im));
            const __m256 product_im =
                _mm256_fmadd_ps(odd_re, rotation_im, _mm256_mul_ps(odd_im, rotation_re));
            _mm256_storeu_ps(real + base + index, _mm256_add_ps(even_re, product_re));
            _mm256_storeu_ps(imag + base + index, _mm256_add_ps(even_im, product_im));
            _mm256_storeu_ps(real + base + half + index, _mm256_sub_ps(even_re, product_re));
            _mm256_storeu_ps(imag + base + half + index, _mm256_sub_ps(even_im, product_im));
        }
    }
}

template <std::size_t Count>
inline void ifft_one(float* const real, float* const imag) noexcept {
    static_assert(Count == 16U || Count == 32U);
    ifft_reorder<Count>(real, imag);
    for (std::size_t base = 0U; base < Count; base += 8U) {
        __m256 values_re = stage2(_mm256_loadu_ps(real + base));
        __m256 values_im = stage2(_mm256_loadu_ps(imag + base));
        stage4(values_re, values_im);
        stage8(values_re, values_im);
        _mm256_storeu_ps(real + base, values_re);
        _mm256_storeu_ps(imag + base, values_im);
    }
    if constexpr (Count == 16U) {
        apply_late_stage<16U>(real, imag);
    } else {
        for (std::size_t base = 0U; base < Count; base += 16U) {
            apply_late_stage<16U>(real + base, imag + base);
        }
        apply_late_stage<32U>(real, imag);
    }
}

template <std::size_t Count>
inline void ifft_batch(float* const real, float* const imag,
                       const std::size_t transform_count, const std::size_t stride) noexcept {
    for (std::size_t transform = 0U; transform < transform_count; ++transform) {
        ifft_one<Count>(real + transform * stride, imag + transform * stride);
    }
}

} // namespace

void Ifft_avx2_fma(float* const real, float* const imag, const std::size_t size,
                   const std::size_t transform_count, const std::size_t stride) noexcept {
    switch (size) {
    case 4U:
        Ifft_generic(real, imag, size, transform_count, stride);
        break;
    case 8U:
        for (std::size_t transform = 0U; transform < transform_count; ++transform) {
            ifft8(real + transform * stride, imag + transform * stride);
        }
        break;
    case 16U:
        ifft_batch<16U>(real, imag, transform_count, stride);
        break;
    case 32U:
        ifft_batch<32U>(real, imag, transform_count, stride);
        break;
    default:
        Ifft_generic(real, imag, size, transform_count, stride);
        break;
    }
}

} // namespace uni::simd::detail
