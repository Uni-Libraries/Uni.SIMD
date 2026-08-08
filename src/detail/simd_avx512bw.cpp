//
// Includes
//

// stdlib
#include <cstring>

// compiler
#include <immintrin.h>

#include "detail/simd_avx512bw.h"
#include "detail/simd_generic.h"



//
// Functions
//

namespace uni::simd::detail {
//
// Pack8_LSB
//

void Pack8_LSB_avx512bw(void* dst, const void* src, size_t len) {
    auto* dst8 = static_cast<uint8_t*>(dst);
    const auto* src8 = static_cast<const uint8_t*>(src);

    size_t i = 0;
    const __m512i ones8 = _mm512_set1_epi8(1);

    // 8 output bytes at a time: 64 input "bit-bytes" -> __mmask64 -> store 8 bytes
    for (; i + 8 <= len; i += 8) {
        const uint8_t* p = src8 + i * 8; // 64 bytes
        __m512i v = _mm512_loadu_si512((const void*)p);
        v = _mm512_and_si512(v, ones8);
        v = _mm512_slli_epi16(v, 7);                // LSB -> MSB
        const __mmask64 m = _mm512_movepi8_mask(v); // VPMOVB2M: MSB per byte -> mask
        auto mm = static_cast<uint64_t>(m);
        std::memcpy(dst8 + i, &mm, 8);
    }

    Pack8_LSB_generic(&dst8[i], &src8[i * 8], len - i);
}


//
// Unpack8_LSB
//

void Unpack8_LSB_avx512bw(void* dst, const void* src, size_t len) {
    auto* dst8 = static_cast<uint8_t*>(dst);
    const auto* src8 = static_cast<const uint8_t*>(src);

    size_t i{};

    for (; i + 8 <= len; i += 8) {
        uint64_t mm;
        std::memcpy(&mm, src8 + i, 8);
        const __m512i v = _mm512_maskz_set1_epi8((__mmask64)mm, 1);
        _mm512_storeu_si512(dst8 + i * 8, v);
    }

    // Tail
    Unpack8_LSB_generic(dst8 + i * 8, src8 + i, len - i);
}

} // namespace uni::simd::detail
