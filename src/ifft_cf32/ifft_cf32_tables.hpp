#pragma once

#include <array>
#include <cstddef>
#include <utility>

namespace uni::simd::detail {

alignas(32) inline constexpr std::array<float, 16U> ifft_twiddle_re{
    1.0f,          0.9807852804f,  0.9238795325f,  0.8314696123f,
    0.7071067812f, 0.5555702330f,  0.3826834324f,  0.1950903220f,
    0.0f,         -0.1950903220f, -0.3826834324f, -0.5555702330f,
   -0.7071067812f,-0.8314696123f, -0.9238795325f, -0.9807852804f,
};

alignas(32) inline constexpr std::array<float, 16U> ifft_twiddle_im{
    0.0f, 0.1950903220f, 0.3826834324f, 0.5555702330f,
    0.7071067812f, 0.8314696123f, 0.9238795325f, 0.9807852804f,
    1.0f, 0.9807852804f, 0.9238795325f, 0.8314696123f,
    0.7071067812f, 0.5555702330f, 0.3826834324f, 0.1950903220f,
};

template <std::size_t Count>
[[nodiscard]] consteval std::array<float, Count / 2U>
make_ifft_stage_twiddles(const std::array<float, 16U>& source) noexcept {
    std::array<float, Count / 2U> result{};
    for (std::size_t index = 0U; index < result.size(); ++index) {
        result[index] = source[index * (32U / Count)];
    }
    return result;
}

template <std::size_t Count>
alignas(32) inline constexpr auto ifft_stage_twiddle_re =
    make_ifft_stage_twiddles<Count>(ifft_twiddle_re);

template <std::size_t Count>
alignas(32) inline constexpr auto ifft_stage_twiddle_im =
    make_ifft_stage_twiddles<Count>(ifft_twiddle_im);

template <std::size_t Count>
[[nodiscard]] consteval std::size_t ifft_reverse_bits(std::size_t value) noexcept {
    std::size_t reversed = 0U;
    for (std::size_t remaining = Count; remaining > 1U; remaining /= 2U) {
        reversed = (reversed << 1U) | (value & 1U);
        value >>= 1U;
    }
    return reversed;
}

template <std::size_t Count, std::size_t Index = 0U>
inline void ifft_reorder(float* const real, float* const imag) noexcept {
    if constexpr (Index < Count) {
        constexpr std::size_t reversed = ifft_reverse_bits<Count>(Index);
        if constexpr (Index < reversed) {
            std::swap(real[Index], real[reversed]);
            std::swap(imag[Index], imag[reversed]);
        }
        ifft_reorder<Count, Index + 1U>(real, imag);
    }
}

} // namespace uni::simd::detail
