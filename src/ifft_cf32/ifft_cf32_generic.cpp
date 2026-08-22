#include "ifft_cf32/ifft_cf32_internal.hpp"
#include "ifft_cf32/ifft_cf32_tables.hpp"

#include <cstddef>

namespace uni::simd::detail {
namespace {

template <std::size_t Count, std::size_t Length, std::size_t Butterfly = 0U>
inline void apply_stage(float* const real, float* const imag) noexcept {
    if constexpr (Butterfly < Count / 2U) {
        constexpr std::size_t half = Length / 2U;
        constexpr std::size_t base = (Butterfly / half) * Length;
        constexpr std::size_t index = Butterfly % half;
        constexpr std::size_t even = base + index;
        constexpr std::size_t odd = even + half;
        constexpr std::size_t twiddle = index * (32U / Length);
        const float odd_re = real[odd];
        const float odd_im = imag[odd];
        float product_re;
        float product_im;
        if constexpr (twiddle == 0U) {
            product_re = odd_re;
            product_im = odd_im;
        } else if constexpr (twiddle == 8U) {
            product_re = -odd_im;
            product_im = odd_re;
        } else {
            product_re = odd_re * ifft_twiddle_re[twiddle] - odd_im * ifft_twiddle_im[twiddle];
            product_im = odd_re * ifft_twiddle_im[twiddle] + odd_im * ifft_twiddle_re[twiddle];
        }
        const float even_re = real[even];
        const float even_im = imag[even];
        real[even] = even_re + product_re;
        imag[even] = even_im + product_im;
        real[odd] = even_re - product_re;
        imag[odd] = even_im - product_im;
        apply_stage<Count, Length, Butterfly + 1U>(real, imag);
    }
}

template <std::size_t Count, std::size_t Length = 2U>
inline void apply_stages(float* const real, float* const imag) noexcept {
    if constexpr (Length <= Count) {
        apply_stage<Count, Length>(real, imag);
        apply_stages<Count, Length * 2U>(real, imag);
    }
}

template <std::size_t Count>
inline void ifft_fixed(float* const real, float* const imag) noexcept {
    ifft_reorder<Count>(real, imag);
    apply_stages<Count>(real, imag);
}

} // namespace

void Ifft_generic(float* const real, float* const imag, const std::size_t size,
                  const std::size_t transform_count, const std::size_t stride) noexcept {
    for (std::size_t transform = 0U; transform < transform_count; ++transform) {
        float* const transform_re = real + transform * stride;
        float* const transform_im = imag + transform * stride;
        switch (size) {
        case 4U:
            ifft_fixed<4U>(transform_re, transform_im);
            break;
        case 8U:
            ifft_fixed<8U>(transform_re, transform_im);
            break;
        case 16U:
            ifft_fixed<16U>(transform_re, transform_im);
            break;
        case 32U:
            ifft_fixed<32U>(transform_re, transform_im);
            break;
        default:
            break;
        }
    }
}

} // namespace uni::simd::detail
