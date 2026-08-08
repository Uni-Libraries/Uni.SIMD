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
    void Invert1_avx512f(void* dst, const void* src, size_t len);
    void Invert8_avx512f(void* dst, const void* src, size_t len);
}
