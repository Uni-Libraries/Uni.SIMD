#include "qpsk_carrier_analyzer_internal.hpp"

#if UNI_SIMD_HAVE_NEON

#include <arm_neon.h>

namespace uni::simd::kernels {

void QpskCarrierAnalyzer_neon(uni_simd_qpsk_carrier_analyzer_t& analyzer, const float* input, const std::size_t count,
                              uni_simd_qpsk_carrier_analyzer_result_t& result) noexcept {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t two = vdupq_n_f32(2.0f);
    const float32x4_t epsilon = vdupq_n_f32(analyzer.magnitude_epsilon);
    const uint32x4_t sign_mask = vdupq_n_u32(0x80000000U);

    std::size_t index = 0U;
    for (; index + 4U <= count; index += 4U) {
        const float32x4x2_t components = vld2q_f32(input + 2U * index);
        const float32x4_t real = components.val[0];
        const float32x4_t imag = components.val[1];
        const float32x4_t power = vfmaq_f32(vmulq_f32(imag, imag), real, real);
        const float32x4_t decision_real_sign = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(one), vandq_u32(vreinterpretq_u32_f32(real), sign_mask)));
        const float32x4_t decision_imag_sign = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(one), vandq_u32(vreinterpretq_u32_f32(imag), sign_mask)));
        const float32x4_t decision_real = vfmaq_f32(vmulq_f32(imag, decision_imag_sign), real, decision_real_sign);
        const float32x4_t decision_imag = vfmsq_f32(vmulq_f32(imag, decision_real_sign), real, decision_imag_sign);

        const uint32x4_t valid = vcgtq_f32(power, epsilon);
        const float32x4_t safe_power = vbslq_f32(valid, power, one);
        const float32x4_t square_real = vdivq_f32(vfmsq_f32(vmulq_f32(real, real), imag, imag), safe_power);
        const float32x4_t square_imag = vdivq_f32(vmulq_f32(two, vmulq_f32(real, imag)), safe_power);
        const float32x4_t fourth_real = vbslq_f32(valid, vfmsq_f32(vmulq_f32(square_real, square_real), square_imag, square_imag), zero);
        const float32x4_t fourth_imag = vbslq_f32(valid, vmulq_f32(two, vmulq_f32(square_real, square_imag)), zero);

        alignas(16) float powers[4];
        alignas(16) float decisions_real[4];
        alignas(16) float decisions_imag[4];
        alignas(16) float fourths_real[4];
        alignas(16) float fourths_imag[4];
        alignas(16) std::uint32_t valid_lanes[4];
        vst1q_f32(powers, power);
        vst1q_f32(decisions_real, decision_real);
        vst1q_f32(decisions_imag, decision_imag);
        vst1q_f32(fourths_real, fourth_real);
        vst1q_f32(fourths_imag, fourth_imag);
        vst1q_u32(valid_lanes, valid);
        std::uint32_t valid_bits = 0U;
        for (std::size_t lane = 0U; lane < 4U; ++lane) {
            valid_bits |= (valid_lanes[lane] != 0U ? 1U : 0U) << lane;
        }
        QpskCarrierAnalyzer_commit_batch(analyzer, result, powers, decisions_real, decisions_imag, fourths_real, fourths_imag, valid_bits);
    }
    QpskCarrierAnalyzer_scalar_accumulate(analyzer, input + 2U * index, count - index, result);
}

} // namespace uni::simd::kernels

#endif
