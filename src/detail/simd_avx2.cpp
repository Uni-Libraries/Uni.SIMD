//
// Includes
//

// stdlib
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

    for (; i + 4U <= len; i += 4U) {
        const __m256 x = _mm256_mul_ps(_mm256_loadu_ps(src + 2U * i), inverse_normalization_ps);
        const __m256 sq = _mm256_mul_ps(x, x);
        const __m256 pair_sum = _mm256_hadd_ps(sq, sq);
        const __m256 packed = _mm256_mul_ps(_mm256_permutevar8x32_ps(pair_sum, _mm256_setr_epi32(0, 1, 4, 5, 0, 0, 0, 0)), output_scale_ps);
        _mm_storeu_ps(dst + i, _mm256_castps256_ps128(packed));
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

    // Process 4 output bytes at a time: 32 input "bit-bytes" -> movemask -> 32-bit -> store 4 bytes
    for (; i + 4 <= len; i += 4) {
        const uint8_t* p = src8 + i * 8; // 32 bytes
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));

        // normalize to 0/1
        v = _mm256_and_si256(v, ones8);

        // move LSB -> MSB because movemask reads MSB of each byte
        v = _mm256_slli_epi16(v, 7);

        auto m = static_cast<uint32_t>(_mm256_movemask_epi8(v)); // bits correspond to bytes 0..31

        // little-endian store: out[i+0] gets bits 0..7, out[i+1] gets 8..15, etc.
        std::memcpy(dst8 + i, &m, 4);
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

void Unpack8_LSB_avx2(void* dst, const void* src, size_t len) {
    auto* dst8 = static_cast<uint8_t*>(dst);
    const auto* src8 = static_cast<const uint8_t*>(src);

    size_t i = 0;

    const __m256i shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bitmask = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, static_cast<char>(128), 1, 2, 4, 8, 16, 32, 64, static_cast<char>(128), 1, 2, 4, 8, 16, 32,
                                             64, static_cast<char>(128), 1, 2, 4, 8, 16, 32, 64, static_cast<char>(128));
    const __m256i zero = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi8(1);

    // 4 input bytes -> 32 output bytes (8 bytes per input byte)
    for (; i + 4 <= len; i += 4) {
        uint32_t w;
        std::memcpy(&w, src8 + i, 4);                       // compilers обычно делают mov
        __m256i y = _mm256_set1_epi32(static_cast<int>(w)); // broadcast 32-bit
        __m256i z = _mm256_shuffle_epi8(y, shuf);           // replicate bytes: b0x8, b1x8, b2x8, b3x8
        z = _mm256_and_si256(z, bitmask);                   // 0 or power-of-two

        // want 0/1: (z != 0) ? 1 : 0
        __m256i is_zero = _mm256_cmpeq_epi8(z, zero);
        __m256i out = _mm256_andnot_si256(is_zero, ones);

        _mm256_storeu_si256((__m256i*)(dst8 + i * 8), out);
    }

    // tail
    Unpack8_LSB_generic(dst8 + i * 8, src8 + i, len - i);
}

//
// Unpack8_MSB
//

void Unpack8_MSB_avx2(void* dst, const void* src, size_t len) {
    auto* dst8 = static_cast<uint8_t*>(dst);
    const auto* src8 = static_cast<const uint8_t*>(src);

    size_t i = 0;

    // Same vector path, but with reversed bitmask order per byte.
    const __m256i shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bitmask = _mm256_setr_epi8(static_cast<char>(128), 64, 32, 16, 8, 4, 2, 1, static_cast<char>(128), 64, 32, 16, 8, 4, 2, 1,
                                             static_cast<char>(128), 64, 32, 16, 8, 4, 2, 1, static_cast<char>(128), 64, 32, 16, 8, 4, 2, 1);
    const __m256i zero = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi8(1);

    // 4 input bytes -> 32 output bytes (8 bytes per input byte)
    for (; i + 4 <= len; i += 4) {
        uint32_t w;
        std::memcpy(&w, src8 + i, 4);
        __m256i y = _mm256_set1_epi32(static_cast<int>(w));
        __m256i z = _mm256_shuffle_epi8(y, shuf);
        z = _mm256_and_si256(z, bitmask);

        __m256i is_zero = _mm256_cmpeq_epi8(z, zero);
        __m256i out = _mm256_andnot_si256(is_zero, ones);

        _mm256_storeu_si256((__m256i*)(dst8 + i * 8), out);
    }

    // tail
    Unpack8_MSB_generic(dst8 + i * 8, src8 + i, len - i);
}



//
// MapQPSK_CF32_U8
//

static inline __m256i soft8_ccsds_from_ps(__m256 x, __m256 gain_ps) {
    const __m256 bias = _mm256_set1_ps(128.0f);
    const __m256 zero = _mm256_set1_ps(0.0f);
    const __m256 max255 = _mm256_set1_ps(255.0f);

    __m256 s = _mm256_sub_ps(bias, _mm256_mul_ps(gain_ps, x));
    const __m256 nan_mask = _mm256_cmp_ps(s, s, _CMP_UNORD_Q);
    s = _mm256_max_ps(s, zero);
    s = _mm256_min_ps(s, max255);
    s = _mm256_blendv_ps(s, bias, nan_mask);

    __m256i i32 = _mm256_cvtps_epi32(s);
    __m256i z = _mm256_setzero_si256();
    __m256i i16 = _mm256_packs_epi32(i32, z);
    __m256i u8 = _mm256_packus_epi16(i16, z);
    return u8; // first 4 bytes of each 128-lane hold values
}

void MapQPSK_CF32_U8_avx2(void* dst, const void* src, size_t len, float gain) {
    const float* incf32 = static_cast<const float*>(src);
    uint8_t* dst8 = static_cast<uint8_t*>(dst);

    const __m256 gain_ps = _mm256_set1_ps(gain);
    size_t i = 0;

    for (; i + 8 <= len; i += 8) {
        __m256 a = _mm256_loadu_ps(incf32 + 2 * i + 0); // I0 Q0 I1 Q1 I2 Q2 I3 Q3
        __m256 b = _mm256_loadu_ps(incf32 + 2 * i + 8); // I4 Q4 I5 Q5 I6 Q6 I7 Q7

        const __m256 tI = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
        const __m256 tQ = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1));

        const __m256 I = _mm256_permutevar8x32_ps(tI, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
        const __m256 Q = _mm256_permutevar8x32_ps(tQ, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));

        __m256i u8I = soft8_ccsds_from_ps(I, gain_ps);
        __m256i u8Q = soft8_ccsds_from_ps(Q, gain_ps);

        const __m128i I0 = _mm256_castsi256_si128(u8I);
        const __m128i I1 = _mm256_extracti128_si256(u8I, 1);
        const __m128i Q0 = _mm256_castsi256_si128(u8Q);
        const __m128i Q1 = _mm256_extracti128_si256(u8Q, 1);

        const __m128i iq0 = _mm_unpacklo_epi8(I0, Q0); // 8 bytes for symbols 0..3
        const __m128i iq1 = _mm_unpacklo_epi8(I1, Q1); // 8 bytes for symbols 4..7

        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst8 + 2 * i + 0), iq0);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst8 + 2 * i + 8), iq1);
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
    const __m128 t = _mm_loadu_ps(taps);
    const __m128 lo = _mm_unpacklo_ps(t, t);
    const __m128 hi = _mm_unpackhi_ps(t, t);
    __m256 out = _mm256_castps128_ps256(lo);
    out = _mm256_insertf128_ps(out, hi, 1);
    return out;
}

[[nodiscard]] inline __m256 mul_accumulate_ps_avx2(__m256 acc, const __m256 a, const __m256 b) noexcept { return _mm256_add_ps(acc, _mm256_mul_ps(a, b)); }

[[nodiscard]] inline std::complex<float> reduce_complex_acc_avx2(__m256 acc) noexcept {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    return {tmp[0] + tmp[2] + tmp[4] + tmp[6], tmp[1] + tmp[3] + tmp[5] + tmp[7]};
}

template <typename MulAccumulate>
[[nodiscard]] inline std::complex<float> dotprod_cf32real_avx2_impl(const float* x, const float* taps, size_t len, MulAccumulate&& mulacc) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 16U <= len; i += 16U) {
        const __m256 xv0 = _mm256_loadu_ps(x + 2U * (i + 0U));
        const __m256 xv1 = _mm256_loadu_ps(x + 2U * (i + 4U));
        const __m256 xv2 = _mm256_loadu_ps(x + 2U * (i + 8U));
        const __m256 xv3 = _mm256_loadu_ps(x + 2U * (i + 12U));

        acc0 = mulacc(acc0, xv0, duplicate_real_taps4_avx2(taps + i + 0U));
        acc1 = mulacc(acc1, xv1, duplicate_real_taps4_avx2(taps + i + 4U));
        acc2 = mulacc(acc2, xv2, duplicate_real_taps4_avx2(taps + i + 8U));
        acc3 = mulacc(acc3, xv3, duplicate_real_taps4_avx2(taps + i + 12U));
    }

    for (; i + 4U <= len; i += 4U) {
        const __m256 xv = _mm256_loadu_ps(x + 2U * i);
        acc0 = mulacc(acc0, xv, duplicate_real_taps4_avx2(taps + i));
    }

    const std::complex<float> vec_acc = reduce_complex_acc_avx2(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));

    float acc_re = vec_acc.real();
    float acc_im = vec_acc.imag();
    for (; i < len; ++i) {
        const float tap = taps[i];
        acc_re += x[2U * i + 0U] * tap;
        acc_im += x[2U * i + 1U] * tap;
    }

    return {acc_re, acc_im};
}

} // namespace

std::complex<float> DotProdCF32Real_avx2(const void* src, const float* taps, size_t len) noexcept {
    if (!src || !taps || len == 0U) {
        return {};
    }
    return dotprod_cf32real_avx2_impl(static_cast<const float*>(src), taps, len, mul_accumulate_ps_avx2);
}


std::complex<float> DotProdSymmetricCF32Real_avx2(const void* src, const float* taps_pairs, size_t pair_count, float center_tap) noexcept {
    std::complex<float> out{};
    if (!src) {
        return out;
    }

    const auto* x = static_cast<const float*>(src);
    const size_t center_off = 2U * pair_count;
    out.real(x[center_off + 0U] * center_tap);
    out.imag(x[center_off + 1U] * center_tap);

    if (pair_count == 0U) {
        return out;
    }

    const size_t tail_off = 4U * pair_count;
    const __m256i reverse_complex4 = _mm256_setr_epi32(6, 7, 4, 5, 2, 3, 0, 1);
    __m256 acc = _mm256_setzero_ps();
    size_t k = 0;

    for (; k + 4U <= pair_count; k += 4U) {
        const __m256 left = _mm256_loadu_ps(x + 2U * k);
        __m256 right = _mm256_loadu_ps(x + (tail_off - (2U * k + 6U)));
        right = _mm256_permutevar8x32_ps(right, reverse_complex4);
        const __m256 pair_sum = _mm256_add_ps(left, right);
        const __m256 tv = duplicate_real_taps4_avx2(taps_pairs + k);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(pair_sum, tv));
    }

    const std::complex<float> vec_acc = reduce_complex_acc_avx2(acc);
    out.real(out.real() + vec_acc.real());
    out.imag(out.imag() + vec_acc.imag());

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
