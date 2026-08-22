#pragma once

#include <cstddef>

namespace uni::simd::detail {

using IfftFn = void (*)(float*, float*, std::size_t) noexcept;
using IfftSupportFn = bool (*)(std::size_t) noexcept;

void Ifft_generic(float* real, float* imag, std::size_t count) noexcept;
void Ifft8_avx2_fma(float* real, float* imag, std::size_t count) noexcept;

[[nodiscard]] constexpr bool Ifft_supports_all(const std::size_t count) noexcept {
    return count == 4U || count == 8U || count == 16U || count == 32U;
}

[[nodiscard]] constexpr bool Ifft_supports_8(const std::size_t count) noexcept {
    return count == 8U;
}

} // namespace uni::simd::detail
