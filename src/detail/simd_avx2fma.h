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

[[nodiscard]] std::complex<float> DotProdSymmetricCF32Real_avx2fma(const void* src, const float* taps_pairs, size_t pair_count, float center_tap) noexcept;

}
