#pragma once

//
// Includes
//

// stdlib
#include <complex>
#include <cstddef>



//
// Functions
//

namespace uni::simd::detail {
void Invert1_sse2(void* dst, const void* src, size_t len);
void Invert8_sse2(void* dst, const void* src, size_t len);
void Pack8_LSB_sse2(void* dst, const void* src, size_t len);
void Pack8_MSB_sse2(void* dst, const void* src, size_t len);
void Unpack8_LSB_sse2(void* dst, const void* src, size_t len);
void Unpack8_MSB_sse2(void* dst, const void* src, size_t len);
void MapQPSK_CF32_U8_sse2(void* dst, const void* src, size_t len, float gain);
void PowerSpectrumCF32F32_sse2(float* dst, const std::complex<float>* src, size_t len, float normalization_factor) noexcept;
void PowerSpectralDensityCF32F32_sse2(float* dst, const std::complex<float>* src, size_t len, float normalization_factor, float rbw_hz) noexcept;
} // namespace uni::simd::detail
