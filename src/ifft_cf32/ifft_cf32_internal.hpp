#pragma once

#include <cstddef>

namespace uni::simd::detail {

void Ifft_generic(float* real, float* imag, std::size_t size,
                  std::size_t transform_count, std::size_t stride) noexcept;
void Ifft_avx2_fma(float* real, float* imag, std::size_t size,
                   std::size_t transform_count, std::size_t stride) noexcept;
void Ifft_neon(float* real, float* imag, std::size_t size,
               std::size_t transform_count, std::size_t stride) noexcept;

[[nodiscard]] constexpr bool Ifft_supports_all(const std::size_t count) noexcept {
    return count == 4U || count == 8U || count == 16U || count == 32U;
}

[[nodiscard]] constexpr bool Ifft_supports_simd(const std::size_t count) noexcept {
    return count == 8U || count == 16U || count == 32U;
}

} // namespace uni::simd::detail
