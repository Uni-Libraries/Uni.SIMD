#pragma once

//
// Includes
//

// stdlib
#include <cstddef>



//
// Functions
//

namespace uni::simd::detail {
    void Pack8_LSB_avx512bw(void* dst, const void* src, size_t len);
    void Unpack8_LSB_avx512bw(void* dst, const void* src, size_t len);
}
