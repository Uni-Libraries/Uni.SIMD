//
// Includes
//

// stdlib
#include <cstdint>
#include <cstring>

// compiler
#include <immintrin.h>

#include "detail/simd_avx2.h"
#include "detail/simd_generic.h"
#include "detail/simd_sse2.h"



//
// Functions
//

namespace uni::simd::detail {

namespace {

void power_spectrum_cf32f32_scalar_tail(float* dst, const float* src, const size_t len, const float inverse_normalization,
                                        const float output_scale) noexcept {
    for (size_t i = 0; i < len; ++i) {
        const float re = src[2U * i + 0U] * inverse_normalization;
        const float im = src[2U * i + 1U] * inverse_normalization;
        dst[i] = (re * re + im * im) * output_scale;
    }
}

void power_spectrum_cf32f32_avx2_impl(float* dst, const float* src, const size_t len, const float inverse_normalization,
                                     const float output_scale) noexcept {
    const __m256 inverse_normalization_ps = _mm256_set1_ps(inverse_normalization);
    const __m256 output_scale_ps = _mm256_set1_ps(output_scale);
    size_t i = 0;

    for (; i + 8U <= len; i += 8U) {
        const __m256 x0 = _mm256_mul_ps(_mm256_loadu_ps(src + 2U * i), inverse_normalization_ps);
        const __m256 x1 = _mm256_mul_ps(_mm256_loadu_ps(src + 2U * i + 8U), inverse_normalization_ps);
        const __m256 sq0 = _mm256_mul_ps(x0, x0);
        const __m256 sq1 = _mm256_mul_ps(x1, x1);
        const __m256 pair_sum = _mm256_hadd_ps(sq0, sq1);
        const __m256 packed = _mm256_castsi256_ps(
            _mm256_permute4x64_epi64(_mm256_castps_si256(pair_sum), _MM_SHUFFLE(3, 1, 2, 0)));
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(packed, output_scale_ps));
    }

    if (i < len) {
        power_spectrum_cf32f32_scalar_tail(dst + i, src + 2U * i, len - i, inverse_normalization, output_scale);
    }
}

} // namespace

//
// Invert1
//

void Invert1_avx2(void* dst, const void* src, size_t len) {
    auto* dst8 = static_cast<uint8_t*>(dst);
    const auto* src8 = static_cast<const uint8_t*>(src);

    const __m256i ones_lsb = _mm256_set1_epi8(0x01);
    size_t i = 0;

    for (; i + 32 <= len; i += 32) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src8 + i));
        v = _mm256_xor_si256(v, ones_lsb);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst8 + i), v);
    }

    Invert1_generic(dst8 + i, src8 + i, len - i);
}



//
// Invert
//

void Invert8_avx2(void* dst, const void* src, size_t len) {
    auto* dst8 = static_cast<uint8_t*>(dst);
    const auto* src8 = static_cast<const uint8_t*>(src);

    const __m256i ones = _mm256_set1_epi8(static_cast<char>(0xFF));
    size_t i = 0;

    for (; i + 32 <= len; i += 32) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src8 + i));
        v = _mm256_xor_si256(v, ones);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst8 + i), v);
    }

    // Tail
    Invert8_generic(dst8 + i, src8 + i, len - i);
}


//
// Pack8_LSB
//

void Pack8_LSB_avx2(void* dst, const void* src, size_t len) {
    auto* dst8 = static_cast<uint8_t*>(dst);
    const auto* src8 = static_cast<const uint8_t*>(src);

    // in_bits length = nbytes * 8
    size_t i = 0;

    const __m256i ones8 = _mm256_set1_epi8(1);

    // Process 8 output bytes through two independent movemask chains.
    for (; i + 8 <= len; i += 8) {
        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src8 + (i + 0) * 8));
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src8 + (i + 4) * 8));

        v0 = _mm256_and_si256(v0, ones8);
        v1 = _mm256_and_si256(v1, ones8);
        v0 = _mm256_slli_epi16(v0, 7);
        v1 = _mm256_slli_epi16(v1, 7);

        const uint32_t m0 = static_cast<uint32_t>(_mm256_movemask_epi8(v0));
        const uint32_t m1 = static_cast<uint32_t>(_mm256_movemask_epi8(v1));
        std::memcpy(dst8 + i + 0, &m0, 4);
        std::memcpy(dst8 + i + 4, &m1, 4);
    }

    if (i + 4 <= len) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src8 + i * 8));
        v = _mm256_and_si256(v, ones8);
        v = _mm256_slli_epi16(v, 7);

        const uint32_t m = static_cast<uint32_t>(_mm256_movemask_epi8(v));
        std::memcpy(dst8 + i, &m, 4);
        i += 4;
    }

    // Tail
    Pack8_LSB_generic(&dst8[i], &src8[i * 8], len - i);
}

//
// Pack8_MSB
//

void Pack8_MSB_avx2(void* dst, const void* src, size_t len) {
    auto* dst8 = static_cast<uint8_t*>(dst);
    const auto* src8 = static_cast<const uint8_t*>(src);

    size_t i = 0;

    const __m256i ones = _mm256_set1_epi8(1);

    // reverse bytes within each 8-byte block, per 128-bit lane
    const __m256i rev8 = _mm256_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);

    // 8 output bytes per iter: 2x (32 input bytes -> movemask -> 4 output bytes)
    for (; i + 8 <= len; i += 8) {
        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src8 + (i + 0) * 8));
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src8 + (i + 4) * 8));

        v0 = _mm256_and_si256(v0, ones);
        v1 = _mm256_and_si256(v1, ones);

        // make movemask bits map directly to output bit positions
        v0 = _mm256_shuffle_epi8(v0, rev8);
        v1 = _mm256_shuffle_epi8(v1, rev8);

        // put bit into MSB of each byte for movemask
        v0 = _mm256_slli_epi16(v0, 7);
        v1 = _mm256_slli_epi16(v1, 7);

        uint32_t m0 = (uint32_t)_mm256_movemask_epi8(v0); // 4 bytes packed in little-endian
        uint32_t m1 = (uint32_t)_mm256_movemask_epi8(v1);

        std::memcpy(dst8 + i + 0, &m0, 4);
        std::memcpy(dst8 + i + 4, &m1, 4);
    }

    // 4 output bytes
    if (i + 4 <= len) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src8 + i * 8));
        v = _mm256_and_si256(v, ones);
        v = _mm256_shuffle_epi8(v, rev8);
        v = _mm256_slli_epi16(v, 7);

        uint32_t m = (uint32_t)_mm256_movemask_epi8(v);
        std::memcpy(dst8 + i, &m, 4);
        i += 4;
    }

    Pack8_MSB_generic(dst8 + i, src8 + i * 8, len - i);
}


//
// Unpack8_LSB
//

namespace {

[[nodiscard]] inline __m256i unpack4_bytes_avx2(const __m256i bytes, const __m256i shuffle,
                                                const __m256i bitmask, const __m256i ones) noexcept {
    return _mm256_min_epu8(_mm256_and_si256(_mm256_shuffle_epi8(bytes, shuffle), bitmask), ones);
}

void unpack8_avx2_impl(void* dst, const void* src, const size_t len, const __m256i bitmask,
                       const bool msb) noexcept {
    auto* dst8 = static_cast<uint8_t*>(dst);
    const auto* src8 = static_cast<const uint8_t*>(src);
    size_t i = 0;

    const __m256i ones = _mm256_set1_epi8(1);
    const __m256i shuffle0 = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                              2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i shuffle1 = _mm256_setr_epi8(4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
                                              6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7);
    const __m256i shuffle2 = _mm256_setr_epi8(8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
                                              10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11);
    const __m256i shuffle3 = _mm256_setr_epi8(12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
                                              14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15);

    for (; i + 16U <= len; i += 16U) {
        const __m256i bytes = _mm256_broadcastsi128_si256(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(src8 + i)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst8 + i * 8U + 0U),
                            unpack4_bytes_avx2(bytes, shuffle0, bitmask, ones));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst8 + i * 8U + 32U),
                            unpack4_bytes_avx2(bytes, shuffle1, bitmask, ones));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst8 + i * 8U + 64U),
                            unpack4_bytes_avx2(bytes, shuffle2, bitmask, ones));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst8 + i * 8U + 96U),
                            unpack4_bytes_avx2(bytes, shuffle3, bitmask, ones));
    }

    if (i + 8U <= len) {
        const __m256i bytes = _mm256_broadcastsi128_si256(
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src8 + i)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst8 + i * 8U + 0U),
                            unpack4_bytes_avx2(bytes, shuffle0, bitmask, ones));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst8 + i * 8U + 32U),
                            unpack4_bytes_avx2(bytes, shuffle1, bitmask, ones));
        i += 8U;
    }

    if (i + 4U <= len) {
        uint32_t w;
        std::memcpy(&w, src8 + i, 4);
        const __m256i bytes = _mm256_set1_epi32(static_cast<int>(w));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst8 + i * 8U),
                            unpack4_bytes_avx2(bytes, shuffle0, bitmask, ones));
        i += 4U;
    }

    if (msb) {
        Unpack8_MSB_generic(dst8 + i * 8U, src8 + i, len - i);
    } else {
        Unpack8_LSB_generic(dst8 + i * 8U, src8 + i, len - i);
    }
}

} // namespace

void Unpack8_LSB_avx2(void* dst, const void* src, const size_t len) {
    const __m256i bitmask = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, static_cast<char>(128),
                                             1, 2, 4, 8, 16, 32, 64, static_cast<char>(128),
                                             1, 2, 4, 8, 16, 32, 64, static_cast<char>(128),
                                             1, 2, 4, 8, 16, 32, 64, static_cast<char>(128));
    unpack8_avx2_impl(dst, src, len, bitmask, false);
}

//
// Unpack8_MSB
//

void Unpack8_MSB_avx2(void* dst, const void* src, const size_t len) {
    const __m256i bitmask = _mm256_setr_epi8(static_cast<char>(128), 64, 32, 16, 8, 4, 2, 1,
                                             static_cast<char>(128), 64, 32, 16, 8, 4, 2, 1,
                                             static_cast<char>(128), 64, 32, 16, 8, 4, 2, 1,
                                             static_cast<char>(128), 64, 32, 16, 8, 4, 2, 1);
    unpack8_avx2_impl(dst, src, len, bitmask, true);
}



//
// MapQPSK_CF32_U8
//

static inline __m256i soft8_ccsds_i32_from_ps(__m256 x, __m256 gain_ps) {
    const __m256 bias = _mm256_set1_ps(128.0f);
    const __m256 zero = _mm256_set1_ps(0.0f);
    const __m256 max255 = _mm256_set1_ps(255.0f);

    __m256 s = _mm256_sub_ps(bias, _mm256_mul_ps(gain_ps, x));
    const __m256 nan_mask = _mm256_cmp_ps(s, s, _CMP_UNORD_Q);
    s = _mm256_max_ps(s, zero);
    s = _mm256_min_ps(s, max255);
    s = _mm256_blendv_ps(s, bias, nan_mask);

    return _mm256_cvtps_epi32(s);
}

void MapQPSK_CF32_U8_avx2(void* dst, const void* src, size_t len, float gain) {
    const float* incf32 = static_cast<const float*>(src);
    uint8_t* dst8 = static_cast<uint8_t*>(dst);

    const __m256 gain_ps = _mm256_set1_ps(gain);
    size_t i = 0;

    for (; i + 8 <= len; i += 8) {
        const __m256 a = _mm256_loadu_ps(incf32 + 2 * i);
        const __m256 b = _mm256_loadu_ps(incf32 + 2 * i + 8);
        const __m256i i32a = soft8_ccsds_i32_from_ps(a, gain_ps);
        const __m256i i32b = soft8_ccsds_i32_from_ps(b, gain_ps);
        const __m256i i16 = _mm256_packs_epi32(i32a, i32b);
        const __m128i u8 = _mm_packus_epi16(_mm256_castsi256_si128(i16), _mm256_extracti128_si256(i16, 1));
        const __m128i packed = _mm_shuffle_epi32(u8, _MM_SHUFFLE(3, 1, 2, 0));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst8 + 2 * i), packed);
    }

    if (i < len) {
        MapQPSK_CF32_U8_sse2(dst8 + 2 * i, incf32 + 2 * i, len - i, gain);
    }
}

void PowerSpectrumCF32F32_avx2(float* dst, const std::complex<float>* src, const size_t len, const float normalization_factor) noexcept {
    if (!dst || !src || len == 0U || !std::isfinite(normalization_factor) || normalization_factor <= 0.0f) {
        return;
    }

    const float inv_norm = 1.0f / normalization_factor;
    power_spectrum_cf32f32_avx2_impl(dst, static_cast<const float*>(static_cast<const void*>(src)), len, inv_norm, 1.0f);
}

void PowerSpectralDensityCF32F32_avx2(float* dst, const std::complex<float>* src, const size_t len, const float normalization_factor,
                                      const float rbw_hz) noexcept {
    if (!dst || !src || len == 0U || !std::isfinite(normalization_factor) || normalization_factor <= 0.0f || !std::isfinite(rbw_hz) || rbw_hz <= 0.0f) {
        return;
    }

    const float component_scale = (1.0f / normalization_factor) / std::sqrt(rbw_hz);
    power_spectrum_cf32f32_avx2_impl(dst, static_cast<const float*>(static_cast<const void*>(src)), len, component_scale, 1.0f);
}



namespace {
[[nodiscard]] inline __m256 duplicate_real_taps4_avx2(const float* taps) noexcept {
    // Match four real taps to the interleaved {real, imag} components of four complex samples.
    const __m128 t = _mm_loadu_ps(taps);
    const __m128 lo = _mm_unpacklo_ps(t, t);
    const __m128 hi = _mm_unpackhi_ps(t, t);
    __m256 out = _mm256_castps128_ps256(lo);
    out = _mm256_insertf128_ps(out, hi, 1);
    return out;
}

} // namespace

std::complex<float> DotProdCF32Real_avx2(const void* src, const float* taps, size_t len) noexcept {
    if (!src || !taps || len == 0U) {
        return {};
    }

    const auto* x = static_cast<const float*>(src);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    size_t i = 0;

    // Use independent accumulators to hide multiply-add latency across groups of 16 samples.
    for (; i + 16U <= len; i += 16U) {
        const __m256 xv0 = _mm256_loadu_ps(x + 2U * (i + 0U));
        const __m256 xv1 = _mm256_loadu_ps(x + 2U * (i + 4U));
        const __m256 xv2 = _mm256_loadu_ps(x + 2U * (i + 8U));
        const __m256 xv3 = _mm256_loadu_ps(x + 2U * (i + 12U));

        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(xv0, duplicate_real_taps4_avx2(taps + i + 0U)));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(xv1, duplicate_real_taps4_avx2(taps + i + 4U)));
        acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(xv2, duplicate_real_taps4_avx2(taps + i + 8U)));
        acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(xv3, duplicate_real_taps4_avx2(taps + i + 12U)));
    }

    // Consume any remaining complete four-sample vectors before the scalar tail.
    for (; i + 4U <= len; i += 4U) {
        const __m256 xv = _mm256_loadu_ps(x + 2U * i);
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(xv, duplicate_real_taps4_avx2(taps + i)));
    }

    // Fold the four interleaved SIMD lanes into one {real, imag} result.
    const __m256 vec_acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    const __m128 lane_sum =
        _mm_add_ps(_mm256_castps256_ps128(vec_acc), _mm256_extractf128_ps(vec_acc, 1));
    const __m128 complex_sum = _mm_add_ps(lane_sum, _mm_movehl_ps(lane_sum, lane_sum));
    float acc_re = _mm_cvtss_f32(complex_sum);
    float acc_im = _mm_cvtss_f32(_mm_shuffle_ps(complex_sum, complex_sum, _MM_SHUFFLE(1, 1, 1, 1)));

    // Finish one to three samples without reading beyond the input ranges.
    for (; i < len; ++i) {
        const float tap = taps[i];
        acc_re += x[2U * i + 0U] * tap;
        acc_im += x[2U * i + 1U] * tap;
    }

    return {acc_re, acc_im};
}


std::complex<float> DotProdSymmetricCF32Real_avx2(const void* src, const float* taps_pairs, size_t pair_count, float center_tap) noexcept {
    std::complex<float> out{};
    if (!src) {
        return out;
    }

    const auto* x = static_cast<const float*>(src);
    const size_t center_off = 2U * pair_count;

    // Seed the result with the center sample, which has no mirrored partner.
    out.real(x[center_off + 0U] * center_tap);
    out.imag(x[center_off + 1U] * center_tap);

    if (pair_count == 0U) {
        return out;
    }

    const size_t tail_off = 4U * pair_count;
    const __m256i reverse_complex4 = _mm256_setr_epi32(6, 7, 4, 5, 2, 3, 0, 1);
    __m256 acc = _mm256_setzero_ps();
    size_t k = 0;

    // Reverse the right side, add mirrored samples, then apply one real tap per complex pair.
    for (; k + 4U <= pair_count; k += 4U) {
        const __m256 left = _mm256_loadu_ps(x + 2U * k);
        __m256 right = _mm256_loadu_ps(x + (tail_off - (2U * k + 6U)));
        right = _mm256_permutevar8x32_ps(right, reverse_complex4);
        const __m256 pair_sum = _mm256_add_ps(left, right);
        const __m256 tv = duplicate_real_taps4_avx2(taps_pairs + k);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(pair_sum, tv));
    }

    // Fold the four accumulated complex pairs into the seeded center result.
    const __m128 lane_sum = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    const __m128 complex_sum = _mm_add_ps(lane_sum, _mm_movehl_ps(lane_sum, lane_sum));
    out.real(out.real() + _mm_cvtss_f32(complex_sum));
    out.imag(out.imag() +
             _mm_cvtss_f32(_mm_shuffle_ps(complex_sum, complex_sum, _MM_SHUFFLE(1, 1, 1, 1))));

    // Finish mirrored pairs that do not fill a complete SIMD vector.
    for (; k < pair_count; ++k) {
        const float tap = taps_pairs[k];
        const size_t lo = 2U * k;
        const size_t hi = 4U * pair_count - 2U * k;
        out.real(out.real() + tap * (x[lo + 0U] + x[hi + 0U]));
        out.imag(out.imag() + tap * (x[lo + 1U] + x[hi + 1U]));
    }

    return out;
}

} // namespace uni::simd::detail
