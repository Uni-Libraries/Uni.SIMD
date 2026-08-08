//
// Includes
//

// stdlib
#include <cstdint>

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

    for (; i + 4U <= len; i += 4U) {
        const __m128 lo = _mm_loadu_ps(src + 2U * i);
        const __m128 hi = _mm_loadu_ps(src + 2U * i + 4U);
        const __m128 re = _mm_mul_ps(_mm_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0)), inverse_normalization_ps);
        const __m128 im = _mm_mul_ps(_mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 1, 3, 1)), inverse_normalization_ps);
        const __m128 magnitude_squared = _mm_add_ps(_mm_mul_ps(re, re), _mm_mul_ps(im, im));
        _mm_storeu_ps(dst + i, _mm_mul_ps(magnitude_squared, output_scale_ps));
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

void Unpack8_LSB_sse2(void* dst, const void* src, size_t len) {
    auto* dst8 = static_cast<uint8_t*>(dst);
    const auto* src8 = static_cast<const uint8_t*>(src);

    const __m128i bit_masks = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, static_cast<char>(128),
                                            1, 2, 4, 8, 16, 32, 64, static_cast<char>(128));
    const __m128i ones = _mm_set1_epi8(1);
    size_t i = 0;

    for (; i + 8 <= len; i += 8) {
        const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src8 + i));
        const __m128i pairs = _mm_unpacklo_epi8(bytes, bytes);
        const __m128i low_quads = _mm_unpacklo_epi16(pairs, pairs);
        const __m128i high_quads = _mm_unpackhi_epi16(pairs, pairs);

        __m128i out0 = _mm_unpacklo_epi32(low_quads, low_quads);
        __m128i out1 = _mm_unpackhi_epi32(low_quads, low_quads);
        __m128i out2 = _mm_unpacklo_epi32(high_quads, high_quads);
        __m128i out3 = _mm_unpackhi_epi32(high_quads, high_quads);

        out0 = _mm_min_epu8(_mm_and_si128(out0, bit_masks), ones);
        out1 = _mm_min_epu8(_mm_and_si128(out1, bit_masks), ones);
        out2 = _mm_min_epu8(_mm_and_si128(out2, bit_masks), ones);
        out3 = _mm_min_epu8(_mm_and_si128(out3, bit_masks), ones);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst8 + i * 8 + 0), out0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst8 + i * 8 + 16), out1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst8 + i * 8 + 32), out2);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst8 + i * 8 + 48), out3);
    }

    Unpack8_LSB_generic(dst8 + i * 8, src8 + i, len - i);
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
