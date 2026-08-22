#include "ifft_cf32/ifft_cf32_internal.hpp"
#include "ifft_cf32/ifft_cf32_tables.hpp"

#include <cstddef>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>

namespace uni::simd::detail {
namespace {

alignas(16) constexpr float stage2_sign[4U]{1.0f, -1.0f, 1.0f, -1.0f};
alignas(16) constexpr float stage4_rotation_re[4U]{1.0f, 0.0f, 1.0f, 0.0f};
alignas(16) constexpr float stage4_rotation_im[4U]{0.0f, 1.0f, 0.0f, 1.0f};
alignas(16) constexpr float stage4_sign[4U]{1.0f, 1.0f, -1.0f, -1.0f};

[[nodiscard]] inline float32x4_t apply_stage2(const float32x4_t value) noexcept {
    return vfmaq_f32(vrev64q_f32(value), vld1q_f32(stage2_sign), value);
}

inline void apply_stage4(float32x4_t& real, float32x4_t& imag) noexcept {
    const float32x2_t even_re_half = vget_low_f32(real);
    const float32x2_t even_im_half = vget_low_f32(imag);
    const float32x2_t odd_re_half = vget_high_f32(real);
    const float32x2_t odd_im_half = vget_high_f32(imag);
    const float32x4_t even_re = vcombine_f32(even_re_half, even_re_half);
    const float32x4_t even_im = vcombine_f32(even_im_half, even_im_half);
    const float32x4_t odd_re = vcombine_f32(odd_re_half, odd_re_half);
    const float32x4_t odd_im = vcombine_f32(odd_im_half, odd_im_half);
    const float32x4_t rotation_re = vld1q_f32(stage4_rotation_re);
    const float32x4_t rotation_im = vld1q_f32(stage4_rotation_im);
    const float32x4_t sign = vld1q_f32(stage4_sign);
    const float32x4_t product_re =
        vfmsq_f32(vmulq_f32(odd_re, rotation_re), odd_im, rotation_im);
    const float32x4_t product_im =
        vfmaq_f32(vmulq_f32(odd_im, rotation_re), odd_re, rotation_im);
    real = vfmaq_f32(even_re, sign, product_re);
    imag = vfmaq_f32(even_im, sign, product_im);
}

template <std::size_t Length, std::size_t Count>
inline void apply_stage(float* const real, float* const imag) noexcept {
    constexpr std::size_t half = Length / 2U;
    for (std::size_t base = 0U; base < Count; base += Length) {
        for (std::size_t index = 0U; index < half; index += 4U) {
            const float32x4_t even_re = vld1q_f32(real + base + index);
            const float32x4_t even_im = vld1q_f32(imag + base + index);
            const float32x4_t odd_re = vld1q_f32(real + base + half + index);
            const float32x4_t odd_im = vld1q_f32(imag + base + half + index);
            const float32x4_t rotation_re = vld1q_f32(ifft_stage_twiddle_re<Length>.data() + index);
            const float32x4_t rotation_im = vld1q_f32(ifft_stage_twiddle_im<Length>.data() + index);
            const float32x4_t product_re =
                vfmsq_f32(vmulq_f32(odd_re, rotation_re), odd_im, rotation_im);
            const float32x4_t product_im =
                vfmaq_f32(vmulq_f32(odd_im, rotation_re), odd_re, rotation_im);
            vst1q_f32(real + base + index, vaddq_f32(even_re, product_re));
            vst1q_f32(imag + base + index, vaddq_f32(even_im, product_im));
            vst1q_f32(real + base + half + index, vsubq_f32(even_re, product_re));
            vst1q_f32(imag + base + half + index, vsubq_f32(even_im, product_im));
        }
    }
}

template <std::size_t Count>
inline void ifft_one(float* const real, float* const imag) noexcept {
    static_assert(Count == 8U || Count == 16U || Count == 32U);
    ifft_reorder<Count>(real, imag);
    for (std::size_t base = 0U; base < Count; base += 4U) {
        float32x4_t values_re = apply_stage2(vld1q_f32(real + base));
        float32x4_t values_im = apply_stage2(vld1q_f32(imag + base));
        apply_stage4(values_re, values_im);
        vst1q_f32(real + base, values_re);
        vst1q_f32(imag + base, values_im);
    }
    apply_stage<8U, Count>(real, imag);
    if constexpr (Count >= 16U) {
        apply_stage<16U, Count>(real, imag);
    }
    if constexpr (Count == 32U) {
        apply_stage<32U, Count>(real, imag);
    }
}

template <std::size_t Count>
inline void ifft_batch(float* const real, float* const imag,
                       const std::size_t transform_count, const std::size_t stride) noexcept {
    for (std::size_t transform = 0U; transform < transform_count; ++transform) {
        ifft_one<Count>(real + transform * stride, imag + transform * stride);
    }
}

} // namespace

void Ifft_neon(float* const real, float* const imag, const std::size_t size,
               const std::size_t transform_count, const std::size_t stride) noexcept {
    switch (size) {
    case 4U:
        Ifft_generic(real, imag, size, transform_count, stride);
        break;
    case 8U:
        ifft_batch<8U>(real, imag, transform_count, stride);
        break;
    case 16U:
        ifft_batch<16U>(real, imag, transform_count, stride);
        break;
    case 32U:
        ifft_batch<32U>(real, imag, transform_count, stride);
        break;
    default:
        Ifft_generic(real, imag, size, transform_count, stride);
        break;
    }
}

} // namespace uni::simd::detail
#endif
