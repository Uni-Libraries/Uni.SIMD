#include "ifft_cf32/ifft_cf32_internal.hpp"

#include <cstddef>
#include <utility>

namespace uni::simd::detail {
namespace {

constexpr float twiddle_re[16U]{
    1.0f,          0.9807852804f,  0.9238795325f,  0.8314696123f,
    0.7071067812f, 0.5555702330f,  0.3826834324f,  0.1950903220f,
    0.0f,         -0.1950903220f, -0.3826834324f, -0.5555702330f,
   -0.7071067812f,-0.8314696123f, -0.9238795325f, -0.9807852804f,
};
constexpr float twiddle_im[16U]{
    0.0f, 0.1950903220f, 0.3826834324f, 0.5555702330f,
    0.7071067812f, 0.8314696123f, 0.9238795325f, 0.9807852804f,
    1.0f, 0.9807852804f, 0.9238795325f, 0.8314696123f,
    0.7071067812f, 0.5555702330f, 0.3826834324f, 0.1950903220f,
};

template <std::size_t Count>
[[nodiscard]] consteval std::size_t reverse_bits(std::size_t value) noexcept {
    std::size_t reversed = 0U;
    for (std::size_t remaining = Count; remaining > 1U; remaining /= 2U) {
        reversed = (reversed << 1U) | (value & 1U);
        value >>= 1U;
    }
    return reversed;
}

template <std::size_t Count, std::size_t Index = 0U>
inline void reorder(float* const real, float* const imag) noexcept {
    if constexpr (Index < Count) {
        constexpr std::size_t reversed = reverse_bits<Count>(Index);
        if constexpr (Index < reversed) {
            std::swap(real[Index], real[reversed]);
            std::swap(imag[Index], imag[reversed]);
        }
        reorder<Count, Index + 1U>(real, imag);
    }
}

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
            product_re = odd_re * twiddle_re[twiddle] - odd_im * twiddle_im[twiddle];
            product_im = odd_re * twiddle_im[twiddle] + odd_im * twiddle_re[twiddle];
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
    reorder<Count>(real, imag);
    apply_stages<Count>(real, imag);
}

} // namespace

void Ifft_generic(float* const real, float* const imag, const std::size_t count) noexcept {
    switch (count) {
    case 4U:
        ifft_fixed<4U>(real, imag);
        break;
    case 8U:
        ifft_fixed<8U>(real, imag);
        break;
    case 16U:
        ifft_fixed<16U>(real, imag);
        break;
    case 32U:
        ifft_fixed<32U>(real, imag);
        break;
    default:
        break;
    }
}

} // namespace uni::simd::detail
