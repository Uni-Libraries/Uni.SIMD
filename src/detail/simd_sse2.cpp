//
// Includes
//

// stdlib
#include <array>
#include <cstring>

// compiler
#include <emmintrin.h>

#include "detail/simd_generic.h"
#include "detail/simd_sse2.h"



//
// Functions
//

namespace uni::simd::detail {

namespace {
[[nodiscard]] static inline uint8_t ReverseBits8(uint8_t v) {
    v = static_cast<uint8_t>(((v & 0xF0u) >> 4) | ((v & 0x0Fu) << 4));
    v = static_cast<uint8_t>(((v & 0xCCu) >> 2) | ((v & 0x33u) << 2));
    v = static_cast<uint8_t>(((v & 0xAAu) >> 1) | ((v & 0x55u) << 1));
    return v;
}

void power_spectrum_cf32f32_scalar_tail(float* dst, const float* src, const size_t len, const float inverse_normalization,
                                        const float output_scale) noexcept {
    for (size_t i = 0; i < len; ++i) {
        const float re = src[2U * i + 0U] * inverse_normalization;
        const float im = src[2U * i + 1U] * inverse_normalization;
        dst[i] = (re * re + im * im) * output_scale;
    }
}

void power_spectrum_cf32f32_sse2_impl(float* dst, const float* src, const size_t len, const float inverse_normalization,
                                     const float output_scale) noexcept {
    const __m128 inverse_normalization_ps = _mm_set1_ps(inverse_normalization);
    const __m128 output_scale_ps = _mm_set1_ps(output_scale);
    size_t i = 0;

    for (; i + 2U <= len; i += 2U) {
        const __m128 x = _mm_mul_ps(_mm_loadu_ps(src + 2U * i), inverse_normalization_ps);
        const __m128 sq = _mm_mul_ps(x, x);
        const __m128 swapped = _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(2, 3, 0, 1));
        const __m128 pair_sum = _mm_add_ps(sq, swapped);
        const __m128 packed = _mm_mul_ps(_mm_unpacklo_ps(pair_sum, _mm_movehl_ps(pair_sum, pair_sum)), output_scale_ps);
        _mm_storel_pi(reinterpret_cast<__m64*>(dst + i), packed);
    }

    if (i < len) {
        power_spectrum_cf32f32_scalar_tail(dst + i, src + 2U * i, len - i, inverse_normalization, output_scale);
    }
}
} // namespace

//
// Invert1
//

void Invert1_sse2(void* dst, const void* src, size_t len) {
    auto* dst8 = static_cast<uint8_t*>(dst);
    const auto* src8 = static_cast<const uint8_t*>(src);

    const __m128i ones_lsb = _mm_set1_epi8(0x01);
    size_t i = 0;

    for (; i + 16 <= len; i += 16) {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src8 + i));
        v = _mm_xor_si128(v, ones_lsb); // flips only bit0 in each byte :contentReference[oaicite:3]{index=3}
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst8 + i), v);
    }

    Invert1_generic(dst8 + i, src8 + i, len - i);
}



//
// Invert8
//

void Invert8_sse2(void* dst, const void* src, size_t len) {
    auto* dst8 = static_cast<uint8_t*>(dst);
    const auto* src8 = static_cast<const uint8_t*>(src);

    const __m128i ones = _mm_set1_epi8(static_cast<char>(0xFF));
    size_t i = 0;

    for (; i + 16 <= len; i += 16) {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src8 + i));
        v = _mm_xor_si128(v, ones); // XOR == invert w/ all-ones :contentReference[oaicite:4]{index=4}
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst8 + i), v);
    }

    Invert8_generic(dst8 + i, src8 + i, len - i);
}



//
// Pack8_LSB
//

void Pack8_LSB_sse2(void* dst, const void* src, size_t len) {
    auto* dst8 = static_cast<uint8_t*>(dst);
    const auto* src8 = static_cast<const uint8_t*>(src);

    size_t i = 0;

    const __m128i ones = _mm_set1_epi8(1);

    // 2 output bytes per iteration:
    // each output byte consumes 8 input "bit-bytes" -> 16 input bytes for 2 outputs
    for (; i + 2 <= len; i += 2) {
        const uint8_t* p = src8 + i * 8; // 16 bytes
        __m128i v = _mm_loadu_si128((const __m128i*)p);

        // normalize to 0/1 in LSB
        v = _mm_and_si128(v, ones);

        // movemask reads MSB of each byte -> shift LSB to MSB
        v = _mm_slli_epi16(v, 7);

        const int m = _mm_movemask_epi8(v); // 16-bit meaningful
        dst8[i + 0] = static_cast<uint8_t>(m & 0xFF);
        dst8[i + 1] = static_cast<uint8_t>((m >> 8) & 0xFF);
    }

    // tail
    Pack8_LSB_generic(&dst8[i], &src8[i * 8], len - i);
}



//
// Unpack8_LSB
//

// 256-entry LUT: each entry is 8 bytes (0/1) packed into a uint64.
// Byte k (0..7) of lut[b] == (b>>k)&1.
alignas(64) inline constexpr std::array<std::uint64_t, 256> kUnpack8_LSBLut = [] {
    std::array<std::uint64_t, 256> t{};
    for (int b = 0; b < 256; ++b) {
        std::uint64_t x = 0;
        for (int k = 0; k < 8; ++k) {
            x |= static_cast<std::uint64_t>((b >> k) & 1u) << (k * 8);
        }
        t[static_cast<std::size_t>(b)] = x;
    }
    return t;
}();

alignas(64) inline constexpr std::array<std::uint64_t, 256> kUnpack8_MSBLut = [] {
    std::array<std::uint64_t, 256> t{};
    for (int b = 0; b < 256; ++b) {
        std::uint64_t x = 0;
        for (int k = 0; k < 8; ++k) {
            x |= static_cast<std::uint64_t>((b >> (7 - k)) & 1u) << (k * 8);
        }
        t[static_cast<std::size_t>(b)] = x;
    }
    return t;
}();

static inline __m128i lut2_to_xmm(uint8_t b0, uint8_t b1) {
    // safe bit-cast via set_epi64x (no aliasing concerns)
    const std::uint64_t lo = kUnpack8_LSBLut[b0];
    const std::uint64_t hi = kUnpack8_LSBLut[b1];
    return _mm_set_epi64x(static_cast<long long>(hi), static_cast<long long>(lo));
}

static inline __m128i lut2_to_xmm_msb(uint8_t b0, uint8_t b1) {
    const std::uint64_t lo = kUnpack8_MSBLut[b0];
    const std::uint64_t hi = kUnpack8_MSBLut[b1];
    return _mm_set_epi64x(static_cast<long long>(hi), static_cast<long long>(lo));
}

void Unpack8_LSB_sse2(void* dst, const void* src, size_t len) {
    std::size_t i = 0;

    auto* dst8 = static_cast<uint8_t*>(dst);
    const auto* src8 = static_cast<const uint8_t*>(src);

    // If dst is 16B-aligned, then dst + (i*8) is 16B-aligned for even i.
    if ((reinterpret_cast<uintptr_t>(dst) & 15u) == 0) {
        // Unroll by 8 input bytes -> 64 output bytes (4 aligned stores)
        for (; i + 8 <= len; i += 8) {
            __m128i v0 = lut2_to_xmm(src8[i + 0], src8[i + 1]);
            __m128i v1 = lut2_to_xmm(src8[i + 2], src8[i + 3]);
            __m128i v2 = lut2_to_xmm(src8[i + 4], src8[i + 5]);
            __m128i v3 = lut2_to_xmm(src8[i + 6], src8[i + 7]);

            _mm_store_si128((__m128i*)(dst8 + (i + 0) * 8), v0);
            _mm_store_si128((__m128i*)(dst8 + (i + 2) * 8), v1);
            _mm_store_si128((__m128i*)(dst8 + (i + 4) * 8), v2);
            _mm_store_si128((__m128i*)(dst8 + (i + 6) * 8), v3);
        }

        // Remaining pairs
        for (; i + 2 <= len; i += 2) {
            __m128i v = lut2_to_xmm(src8[i], src8[i + 1]);
            _mm_store_si128((__m128i*)(dst8 + i * 8), v);
        }
    } else {
        // Unaligned stores
        for (; i + 8 <= len; i += 8) {
            const __m128i v0 = lut2_to_xmm(src8[i + 0], src8[i + 1]);
            const __m128i v1 = lut2_to_xmm(src8[i + 2], src8[i + 3]);
            const __m128i v2 = lut2_to_xmm(src8[i + 4], src8[i + 5]);
            const __m128i v3 = lut2_to_xmm(src8[i + 6], src8[i + 7]);

            _mm_storeu_si128((__m128i*)(dst8 + (i + 0) * 8), v0);
            _mm_storeu_si128((__m128i*)(dst8 + (i + 2) * 8), v1);
            _mm_storeu_si128((__m128i*)(dst8 + (i + 4) * 8), v2);
            _mm_storeu_si128((__m128i*)(dst8 + (i + 6) * 8), v3);
        }
        for (; i + 2 <= len; i += 2) {
            const __m128i v = lut2_to_xmm(src8[i], src8[i + 1]);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst8 + i * 8), v);
        }
    }

    // Tail (0 or 1 byte left)
    if (i < len) {
        std::uint64_t x = kUnpack8_LSBLut[src8[i]];
        std::memcpy(dst8 + i * 8, &x, 8);
        ++i;
    }
}

//
// MapQPSK_CF32_U8
//

static inline __m128i soft4_ccsds_from_ps(__m128 x, __m128 gain_ps) {
    // s = 128 - gain*x  (CCSDS polarity)
    const __m128 bias = _mm_set1_ps(128.0f);
    const __m128 zero = _mm_set1_ps(0.0f);
    const __m128 max255 = _mm_set1_ps(255.0f);

    __m128 s = _mm_sub_ps(bias, _mm_mul_ps(gain_ps, x));
    const __m128 nan_mask = _mm_cmpunord_ps(s, s);
    s = _mm_max_ps(s, zero);
    s = _mm_min_ps(s, max255);
    s = _mm_or_ps(_mm_and_ps(nan_mask, bias), _mm_andnot_ps(nan_mask, s));

    const __m128i i32 = _mm_cvtps_epi32(s);
    const __m128i z = _mm_setzero_si128();
    const __m128i i16 = _mm_packs_epi32(i32, z);
    __m128i u8 = _mm_packus_epi16(i16, z);
    return u8; // low 4 bytes valid
}

void MapQPSK_CF32_U8_sse2(void* dst, const void* src, size_t len, float gain) {
    const float* incf32 = static_cast<const float*>(src);
    uint8_t* dst8 = static_cast<uint8_t*>(dst);

    const __m128 gain_ps = _mm_set1_ps(gain);
    size_t i = 0;

    for (; i + 4 <= len; i += 4) {
        __m128 a = _mm_loadu_ps(incf32 + 2 * i + 0); // I0 Q0 I1 Q1
        __m128 b = _mm_loadu_ps(incf32 + 2 * i + 4); // I2 Q2 I3 Q3

        __m128 I = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0)); // I0 I1 I2 I3
        __m128 Q = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1)); // Q0 Q1 Q2 Q3

        __m128i u8I = soft4_ccsds_from_ps(I, gain_ps);
        __m128i u8Q = soft4_ccsds_from_ps(Q, gain_ps);

        const __m128i iq8 = _mm_unpacklo_epi8(u8I, u8Q); // I0 Q0 I1 Q1 I2 Q2 I3 Q3
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst8 + 2 * i), iq8);
    }

    if (i < len) {
        MapQPSK_CF32_U8_generic(dst8 + 2 * i, incf32 + 2 * i, len - i, gain);
    }
}

void PowerSpectrumCF32F32_sse2(float* dst, const std::complex<float>* src, const size_t len, const float normalization_factor) noexcept {
    if (!dst || !src || len == 0U || !std::isfinite(normalization_factor) || normalization_factor <= 0.0f) {
        return;
    }

    const float inv_norm = 1.0f / normalization_factor;
    power_spectrum_cf32f32_sse2_impl(dst, static_cast<const float*>(static_cast<const void*>(src)), len, inv_norm, 1.0f);
}

void PowerSpectralDensityCF32F32_sse2(float* dst, const std::complex<float>* src, const size_t len, const float normalization_factor,
                                      const float rbw_hz) noexcept {
    if (!dst || !src || len == 0U || !std::isfinite(normalization_factor) || normalization_factor <= 0.0f || !std::isfinite(rbw_hz) || rbw_hz <= 0.0f) {
        return;
    }

    const float component_scale = (1.0f / normalization_factor) / std::sqrt(rbw_hz);
    power_spectrum_cf32f32_sse2_impl(dst, static_cast<const float*>(static_cast<const void*>(src)), len, component_scale, 1.0f);
}

} // namespace uni::simd::detail
