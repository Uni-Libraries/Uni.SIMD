#include "pfb_channelizer/pfb_channelizer_internal.hpp"
#include "ifft_cf32/ifft_cf32_internal.hpp"

#include <array>
#include <cstddef>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>

namespace uni::simd::detail {
namespace {

[[nodiscard]] inline float32x4_t reverse_lanes(const float32x4_t value) noexcept {
    return vrev64q_f32(vextq_f32(value, value, 2));
}

} // namespace

std::size_t PfbChannelizer_neon(PfbChannelizerData& data, const PfbChannelizerBlock& block) noexcept {
    const auto& plan = data;
    auto& state = data;

    constexpr std::size_t bins = 8U;
    constexpr std::size_t decimation = 4U;
    const std::size_t rows = PfbChannelizerAccess::rows(plan);
    const std::size_t history_size = PfbChannelizerAccess::history_size(plan);
    const std::size_t history_mask = history_size - 1U;
    const std::size_t selected_count = plan.selected_output_count();
    const std::size_t phase_period = PfbChannelizerAccess::phase_period(plan);
    const float* coefficients = PfbChannelizerAccess::reversed_coefficients(plan);
    const float* rotations_re = PfbChannelizerAccess::branch_rotation_re(plan);
    const float* rotations_im = PfbChannelizerAccess::branch_rotation_im(plan);
    const float32x4_t rotation_re0 = vld1q_f32(rotations_re);
    const float32x4_t rotation_re1 = vld1q_f32(rotations_re + 4U);
    const float32x4_t rotation_im0 = vld1q_f32(rotations_im);
    const float32x4_t rotation_im1 = vld1q_f32(rotations_im + 4U);
    float* history_i = PfbChannelizerAccess::history_i(state);
    float* history_q = PfbChannelizerAccess::history_q(state);
    std::size_t cursor = PfbChannelizerAccess::cursor(state);
    std::size_t decimation_phase = PfbChannelizerAccess::decimation_phase(state);
    std::size_t post_phase = PfbChannelizerAccess::post_phase(state);
    std::size_t produced = 0U;

    alignas(16) std::array<float, pfb_channelizer_max_bins> fft_re{};
    alignas(16) std::array<float, pfb_channelizer_max_bins> fft_im{};

    for (const auto sample : block.input) {
        const float sample_re = sample.real();
        const float sample_im = sample.imag();
        history_i[cursor] = sample_re;
        history_i[cursor + history_size] = sample_re;
        history_q[cursor] = sample_im;
        history_q[cursor + history_size] = sample_im;

        if (decimation_phase == 0U) {
            if (selected_count != 0U) {
                float32x4_t accumulator_re0 = vdupq_n_f32(0.0f);
                float32x4_t accumulator_re1 = vdupq_n_f32(0.0f);
                float32x4_t accumulator_im0 = vdupq_n_f32(0.0f);
                float32x4_t accumulator_im1 = vdupq_n_f32(0.0f);
                for (std::size_t row = 0U; row < rows; ++row) {
                    const float32x4_t coefficient0 = vld1q_f32(coefficients + row * bins);
                    const float32x4_t coefficient1 = vld1q_f32(coefficients + row * bins + 4U);
                    const std::size_t first_sample = cursor + history_size - row * bins - (bins - 1U);
                    const float32x4_t samples_re0 = vld1q_f32(history_i + first_sample);
                    const float32x4_t samples_re1 = vld1q_f32(history_i + first_sample + 4U);
                    const float32x4_t samples_im0 = vld1q_f32(history_q + first_sample);
                    const float32x4_t samples_im1 = vld1q_f32(history_q + first_sample + 4U);
                    accumulator_re0 = vfmaq_f32(accumulator_re0, samples_re0, coefficient0);
                    accumulator_re1 = vfmaq_f32(accumulator_re1, samples_re1, coefficient1);
                    accumulator_im0 = vfmaq_f32(accumulator_im0, samples_im0, coefficient0);
                    accumulator_im1 = vfmaq_f32(accumulator_im1, samples_im1, coefficient1);
                }

                const float32x4_t natural_re0 = reverse_lanes(accumulator_re1);
                const float32x4_t natural_re1 = reverse_lanes(accumulator_re0);
                const float32x4_t natural_im0 = reverse_lanes(accumulator_im1);
                const float32x4_t natural_im1 = reverse_lanes(accumulator_im0);
                const float32x4_t rotated_re0 =
                    vfmsq_f32(vmulq_f32(natural_re0, rotation_re0), natural_im0, rotation_im0);
                const float32x4_t rotated_re1 =
                    vfmsq_f32(vmulq_f32(natural_re1, rotation_re1), natural_im1, rotation_im1);
                const float32x4_t rotated_im0 =
                    vfmaq_f32(vmulq_f32(natural_im0, rotation_re0), natural_re0, rotation_im0);
                const float32x4_t rotated_im1 =
                    vfmaq_f32(vmulq_f32(natural_im1, rotation_re1), natural_re1, rotation_im1);
                vst1q_f32(fft_re.data(), rotated_re0);
                vst1q_f32(fft_re.data() + 4U, rotated_re1);
                vst1q_f32(fft_im.data(), rotated_im0);
                vst1q_f32(fft_im.data() + 4U, rotated_im1);

                Ifft_generic(fft_re.data(), fft_im.data(), bins);
                const auto logical_bins = plan.logical_bins();
                for (std::size_t output = 0U; output < selected_count; ++output) {
                    const std::int32_t logical_bin = logical_bins[output];
                    const std::size_t fft_bin = static_cast<std::size_t>(logical_bin < 0 ? logical_bin + 8 : logical_bin);
                    block.outputs[output][produced] =
                        pfb_apply_post_phase(plan, output, post_phase, fft_re[fft_bin], fft_im[fft_bin]);
                }
            }
            ++produced;
            post_phase = post_phase + 1U == phase_period ? 0U : post_phase + 1U;
        }

        cursor = (cursor + 1U) & history_mask;
        decimation_phase = decimation_phase + 1U == decimation ? 0U : decimation_phase + 1U;
    }

    PfbChannelizerAccess::set_cursor(state, cursor);
    PfbChannelizerAccess::set_decimation_phase(state, decimation_phase);
    PfbChannelizerAccess::set_post_phase(state, post_phase);
    return produced;
}

} // namespace uni::simd::detail
#else
namespace uni::simd::detail {

std::size_t PfbChannelizer_neon(PfbChannelizerData& data, const PfbChannelizerBlock& block) noexcept {
    return PfbChannelizer_generic(data, block);
}

} // namespace uni::simd::detail
#endif
