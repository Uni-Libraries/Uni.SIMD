#pragma once

//
// Includes
//

// stdlib
#include <cstddef>

#include <complex>



//
// Functions
//

namespace uni::simd::detail {
void Invert1_generic(void* dst, const void* src, size_t len);
void Invert8_generic(void* dst, const void* src, size_t len);
void Pack8_LSB_generic(void* dst, const void* src, size_t len);
void Pack8_MSB_generic(void* dst, const void* src, size_t len);
void Unpack8_LSB_generic(void* dst, const void* src, size_t len);
void Unpack8_MSB_generic(void* dst, const void* src, size_t len);
void MapQPSK_CF32_U8_generic(void* dst, const void* src, size_t len, float gain);
void PowerSpectrumCF32F32_generic(float* dst, const void* src, size_t len, float normalization_factor) noexcept;
void PowerSpectralDensityCF32F32_generic(float* dst, const void* src, size_t len, float normalization_factor, float rbw_hz) noexcept;
void PowerSpectrumCF32F32_deterministic(float* dst, const void* src, size_t len, float normalization_factor) noexcept;
void PowerSpectralDensityCF32F32_deterministic(float* dst, const void* src, size_t len, float normalization_factor, float rbw_hz) noexcept;
[[nodiscard]] std::complex<float> DotProdCF32Real_generic(const void* src, const float* taps, size_t len) noexcept;
[[nodiscard]] std::complex<float> DotProdSymmetricCF32Real_generic(const void* src, const float* taps_pairs, size_t pair_count, float center_tap) noexcept;
} // namespace uni::simd::detail
