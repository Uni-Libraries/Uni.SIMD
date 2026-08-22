//
// Includes
//

// stdlib
#include <cstdint>
#include <cstring>

// compiler
#include <immintrin.h>

#include "detail/simd_generic.h"
#include "detail/simd_avx512f.h"



//
// Functions
//

namespace uni::simd::detail
{
    //
    // Invert1
    //

    void Invert1_avx512f(void* dst, const void* src, size_t len)
    {
        auto* dst8 = static_cast<uint8_t*>(dst);
        const auto* src8 = static_cast<const uint8_t*>(src);

        const __m512i ones_lsb = _mm512_set1_epi8(0x01);
        size_t i = 0;

        for (; i + 64 <= len; i += 64)
        {
            __m512i v = _mm512_loadu_si512(src8 + i);
            v = _mm512_xor_si512(v, ones_lsb);
            _mm512_storeu_si512(dst8 + i, v);
        }

        Invert1_generic(dst8 + i, src8 + i, len - i);
    }



    //
    // Invert8
    //

void Invert8_avx512f(void* dst, const void* src, size_t len) {
        auto* dst8 = static_cast<uint8_t*>(dst);
        const auto* src8 = static_cast<const uint8_t*>(src);

        const __m512i ones = _mm512_set1_epi8(static_cast<char>(0xFF));
        size_t i = 0;

        for (; i + 64 <= len; i += 64) {
            __m512i v = _mm512_loadu_si512(src8 + i);
            v = _mm512_xor_si512(v, ones);
            _mm512_storeu_si512(dst8 + i, v);
        }

        // Tail
        Invert8_generic(dst8 + i, src8 + i, len - i);
    }
}
