//
// Includes
//

// stdlib
#include <cstdint>
#include <cstring>

#include "detail/simd_generic.h"

#include <array>
#include <cmath>



//
// Functions
//

namespace uni::simd::detail {

namespace {

void power_spectrum_cf32f32_generic_impl(float* dst, const float* src, const size_t len, const float inverse_normalization,
                                         const float output_scale) noexcept {
    for (size_t i = 0; i < len; ++i) {
        const float re = src[2U * i + 0U] * inverse_normalization;
        const float im = src[2U * i + 1U] * inverse_normalization;
        dst[i] = (re * re + im * im) * output_scale;
    }
}

} // namespace

//
// Invert1
//

void Invert1_generic(void* dst, const void* src, size_t len) {
    size_t i = 0;

    auto* dst8 = static_cast<uint8_t*>(dst);
    const auto* src8 = static_cast<const uint8_t*>(src);

    // XOR each byte with 0x01 => flips only LSB
    constexpr uint64_t k01 = 0x0101010101010101ull;

    for (; i + sizeof(uint64_t) <= len; i += sizeof(uint64_t)) {
        uint64_t v;
        std::memcpy(&v, src8 + i, sizeof(v));
        v ^= k01; // flip bit0 of each byte
        std::memcpy(dst8 + i, &v, sizeof(v));
    }

    for (; i < len; ++i) {
        dst8[i] = static_cast<uint8_t>(src8[i] ^ 0x01u);
    }
}



//
// Invert8
//

void Invert8_generic(void* dst, const void* src, size_t len) {
    size_t i = 0;

    auto* dst8 = static_cast<uint8_t*>(dst);
    const auto* src8 = static_cast<const uint8_t*>(src);

    for (; i + sizeof(uint64_t) <= len; i += sizeof(uint64_t)) {
        uint64_t v;
        std::memcpy(&v, src8 + i, sizeof(v));
        v = ~v;
        std::memcpy(dst8 + i, &v, sizeof(v));
    }

    for (; i < len; ++i) {
        dst8[i] = static_cast<uint8_t>(src8[i] ^ 0xFFu);
    }
}


//
// Pack8_LSB
//

void Pack8_LSB_generic(void* dst, const void* src, size_t len) {
    auto* dst8 = static_cast<uint8_t*>(dst);
    const auto* src8 = static_cast<const uint8_t*>(src);

    for (size_t i = 0; i < len; i++) {
        dst8[i] =
            static_cast<uint8_t>((src8[i * 8 + 0] & 1u) | ((src8[i * 8 + 1] & 1u) << 1) | ((src8[i * 8 + 2] & 1u) << 2) | ((src8[i * 8 + 3] & 1u) << 3) |
                                 ((src8[i * 8 + 4] & 1u) << 4) | ((src8[i * 8 + 5] & 1u) << 5) | ((src8[i * 8 + 6] & 1u) << 6) | ((src8[i * 8 + 7] & 1u) << 7));
    }
}

//
// Pack8_MSB
//

static inline uint8_t Pack8_MSB_1byte_mul(const uint8_t* p8) {
    return static_cast<uint8_t>(((p8[0] & 1U) << 7U) | ((p8[1] & 1U) << 6U) | ((p8[2] & 1U) << 5U) | ((p8[3] & 1U) << 4U) |
                                ((p8[4] & 1U) << 3U) | ((p8[5] & 1U) << 2U) | ((p8[6] & 1U) << 1U) | (p8[7] & 1U));
}

void Pack8_MSB_generic(void* dst, const void* src, size_t len) {
    auto* dst8 = static_cast<uint8_t*>(dst);
    const auto* src8 = static_cast<const uint8_t*>(src);

    for (size_t i = 0; i < len; i++) {
        dst8[i] = Pack8_MSB_1byte_mul(src8 + i * 8);
    }
}



//
// Unpack8_LSB
//

void Unpack8_LSB_generic(void* dst, const void* src, size_t len) {
    auto* dst8 = static_cast<uint8_t*>(dst);
    const auto* src8 = static_cast<const uint8_t*>(src);

    for (size_t i = 0; i < len; i++) {
        dst8[8 * i + 0] = static_cast<uint8_t>((src8[i] >> 0) & 1u);
        dst8[8 * i + 1] = static_cast<uint8_t>((src8[i] >> 1) & 1u);
        dst8[8 * i + 2] = static_cast<uint8_t>((src8[i] >> 2) & 1u);
        dst8[8 * i + 3] = static_cast<uint8_t>((src8[i] >> 3) & 1u);
        dst8[8 * i + 4] = static_cast<uint8_t>((src8[i] >> 4) & 1u);
        dst8[8 * i + 5] = static_cast<uint8_t>((src8[i] >> 5) & 1u);
        dst8[8 * i + 6] = static_cast<uint8_t>((src8[i] >> 6) & 1u);
        dst8[8 * i + 7] = static_cast<uint8_t>((src8[i] >> 7) & 1u);
    }
}

//
// Unpack8_MSB
//

void Unpack8_MSB_generic(void* dst, const void* src, size_t len) {
    auto* dst8 = static_cast<uint8_t*>(dst);
    const auto* src8 = static_cast<const uint8_t*>(src);

    for (size_t i = 0; i < len; i++) {
        dst8[8 * i + 0] = static_cast<uint8_t>((src8[i] >> 7) & 1u);
        dst8[8 * i + 1] = static_cast<uint8_t>((src8[i] >> 6) & 1u);
        dst8[8 * i + 2] = static_cast<uint8_t>((src8[i] >> 5) & 1u);
        dst8[8 * i + 3] = static_cast<uint8_t>((src8[i] >> 4) & 1u);
        dst8[8 * i + 4] = static_cast<uint8_t>((src8[i] >> 3) & 1u);
        dst8[8 * i + 5] = static_cast<uint8_t>((src8[i] >> 2) & 1u);
        dst8[8 * i + 6] = static_cast<uint8_t>((src8[i] >> 1) & 1u);
        dst8[8 * i + 7] = static_cast<uint8_t>((src8[i] >> 0) & 1u);
    }
}


//
// MapQPSK_CF32_U8
//

static inline uint8_t sat_u8_int(int v) {
    if (v < 0)
        return 0;
    if (v > 255)
        return 255;
    return static_cast<uint8_t>(v);
}

static inline uint8_t soft_ccsds_from_component(float x, float gain) {
    const float s = 128.0f - gain * x;
    if (std::isnan(s)) {
        return 128U;
    }
    if (s <= 0.0f) {
        return 0U;
    }
    if (s >= 255.0f) {
        return 255U;
    }
    const int v = static_cast<int>(lrintf(s));
    return sat_u8_int(v);
}

void MapQPSK_CF32_U8_generic(void* dst, const void* src, size_t len, float gain) {
    const float* incf32 = static_cast<const float*>(src);
    uint8_t* dst8 = static_cast<uint8_t*>(dst);

    for (size_t i = 0; i < len; ++i) {
        float I = incf32[2 * i + 0];
        float Q = incf32[2 * i + 1];
        dst8[2 * i + 0] = soft_ccsds_from_component(I, gain);
        dst8[2 * i + 1] = soft_ccsds_from_component(Q, gain);
    }
}

void PowerSpectrumCF32F32_generic(float* dst, const std::complex<float>* src, const size_t len, const float normalization_factor) noexcept {
    if (!dst || !src || len == 0U || !std::isfinite(normalization_factor) || normalization_factor <= 0.0f) {
        return;
    }

    const float inv_norm = 1.0f / normalization_factor;
    power_spectrum_cf32f32_generic_impl(dst, static_cast<const float*>(static_cast<const void*>(src)), len, inv_norm, 1.0f);
}

void PowerSpectralDensityCF32F32_generic(float* dst, const std::complex<float>* src, const size_t len, const float normalization_factor,
                                         const float rbw_hz) noexcept {
    if (!dst || !src || len == 0U || !std::isfinite(normalization_factor) || normalization_factor <= 0.0f || !std::isfinite(rbw_hz) || rbw_hz <= 0.0f) {
        return;
    }

    const float component_scale = (1.0f / normalization_factor) / std::sqrt(rbw_hz);
    power_spectrum_cf32f32_generic_impl(dst, static_cast<const float*>(static_cast<const void*>(src)), len, component_scale, 1.0f);
}



std::complex<float> DotProdCF32Real_generic(const void* src, const float* taps, size_t len) noexcept {
    if (!src || !taps || len == 0U) {
        return {};
    }

    const auto* x = static_cast<const float*>(src);
    float acc_re0 = 0.0f;
    float acc_im0 = 0.0f;
    float acc_re1 = 0.0f;
    float acc_im1 = 0.0f;
    size_t i = 0;

    for (; i + 1U < len; i += 2U) {
        const float tap0 = taps[i + 0U];
        const float tap1 = taps[i + 1U];
        acc_re0 += x[2U * (i + 0U) + 0U] * tap0;
        acc_im0 += x[2U * (i + 0U) + 1U] * tap0;
        acc_re1 += x[2U * (i + 1U) + 0U] * tap1;
        acc_im1 += x[2U * (i + 1U) + 1U] * tap1;
    }

    for (; i < len; ++i) {
        const float tap = taps[i];
        acc_re0 += x[2U * i + 0U] * tap;
        acc_im0 += x[2U * i + 1U] * tap;
    }

    return {acc_re0 + acc_re1, acc_im0 + acc_im1};
}

std::complex<float> DotProdSymmetricCF32Real_generic(const void* src, const float* taps_pairs, size_t pair_count, float center_tap) noexcept {
    std::complex<float> out{};
    if (!src || (pair_count != 0U && !taps_pairs)) {
        return out;
    }

    const auto* x = static_cast<const float*>(src);
    const size_t center_off = 2U * pair_count;
    float acc_re = x[center_off + 0U] * center_tap;
    float acc_im = x[center_off + 1U] * center_tap;
    float pair_re0 = 0.0f;
    float pair_im0 = 0.0f;
    float pair_re1 = 0.0f;
    float pair_im1 = 0.0f;
    float pair_re2 = 0.0f;
    float pair_im2 = 0.0f;
    float pair_re3 = 0.0f;
    float pair_im3 = 0.0f;
    size_t k = 0;

    for (; k + 4U <= pair_count; k += 4U) {
        const size_t hi = 4U * pair_count - 2U * k;
        pair_re0 += taps_pairs[k + 0U] * (x[2U * k + 0U] + x[hi + 0U]);
        pair_im0 += taps_pairs[k + 0U] * (x[2U * k + 1U] + x[hi + 1U]);
        pair_re1 += taps_pairs[k + 1U] * (x[2U * k + 2U] + x[hi - 2U]);
        pair_im1 += taps_pairs[k + 1U] * (x[2U * k + 3U] + x[hi - 1U]);
        pair_re2 += taps_pairs[k + 2U] * (x[2U * k + 4U] + x[hi - 4U]);
        pair_im2 += taps_pairs[k + 2U] * (x[2U * k + 5U] + x[hi - 3U]);
        pair_re3 += taps_pairs[k + 3U] * (x[2U * k + 6U] + x[hi - 6U]);
        pair_im3 += taps_pairs[k + 3U] * (x[2U * k + 7U] + x[hi - 5U]);
    }

    acc_re += ((pair_re0 + pair_re1) + pair_re2) + pair_re3;
    acc_im += ((pair_im0 + pair_im1) + pair_im2) + pair_im3;

    for (; k < pair_count; ++k) {
        const float tap = taps_pairs[k];
        const size_t lo = 2U * k;
        const size_t hi = 4U * pair_count - 2U * k;
        acc_re += tap * (x[lo + 0U] + x[hi + 0U]);
        acc_im += tap * (x[lo + 1U] + x[hi + 1U]);
    }

    out.real(acc_re);
    out.imag(acc_im);
    return out;
}

} // namespace uni::simd::detail
