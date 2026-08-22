#include "pfb_channelizer/pfb_channelizer_internal.hpp"
#include "ifft_cf32/ifft_cf32_internal.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>

namespace uni::simd::detail {
namespace {

[[nodiscard]] inline float32x4_t reverse_lanes(const float32x4_t value) noexcept {
    return vrev64q_f32(vextq_f32(value, value, 2));
}

[[nodiscard]] inline float horizontal_sum(const float32x4_t value) noexcept {
    const float32x2_t halves = vadd_f32(vget_low_f32(value), vget_high_f32(value));
    return vget_lane_f32(vpadd_f32(halves, halves), 0);
}

template <std::size_t Bins, std::size_t HopCount, bool RowIlp = false>
void process_batch(const PfbChannelizerData& data, const PfbChannelizerBlock& block,
                   const std::size_t* const cursors, const std::size_t* const phases,
                   const std::size_t output_index) noexcept {
    static_assert(Bins % 4U == 0U);
    constexpr std::size_t width = 4U;
    constexpr std::size_t chunk_count = Bins / width;
    const std::size_t rows = PfbChannelizerAccess::rows(data);
    const std::size_t history_size = PfbChannelizerAccess::history_size(data);
    const float* coefficients = PfbChannelizerAccess::reversed_coefficients(data);
    const float* history_i = PfbChannelizerAccess::history_i(data);
    const float* history_q = PfbChannelizerAccess::history_q(data);
    const float* rotations_re = PfbChannelizerAccess::branch_rotation_re(data);
    const float* rotations_im = PfbChannelizerAccess::branch_rotation_im(data);
    const bool direct = data.selected_output_count() == 1U;
    const float* weights_re = PfbChannelizerAccess::selected_transform_re(data);
    const float* weights_im = PfbChannelizerAccess::selected_transform_im(data);
    alignas(16) std::array<float, 4U * Bins> values_re;
    alignas(16) std::array<float, 4U * Bins> values_im;
    float32x4_t direct_re[4U];
    float32x4_t direct_im[4U];
    if (direct) {
        for (std::size_t hop = 0U; hop < HopCount; ++hop) {
            direct_re[hop] = vdupq_n_f32(0.0f);
            direct_im[hop] = vdupq_n_f32(0.0f);
        }
    }

    for (std::size_t destination_chunk = 0U; destination_chunk < chunk_count; ++destination_chunk) {
        const std::size_t source_chunk = chunk_count - 1U - destination_chunk;
        constexpr std::size_t chain_count = RowIlp ? 4U : 1U;
        float32x4_t accumulator_re[4U][4U];
        float32x4_t accumulator_im[4U][4U];
        for (std::size_t hop = 0U; hop < HopCount; ++hop) {
            for (std::size_t chain = 0U; chain < chain_count; ++chain) {
                accumulator_re[hop][chain] = vdupq_n_f32(0.0f);
                accumulator_im[hop][chain] = vdupq_n_f32(0.0f);
            }
        }
        for (std::size_t row = 0U; row < rows; ++row) {
            const std::size_t chunk_offset = source_chunk * width;
            const float32x4_t coefficient = vld1q_f32(coefficients + row * Bins + chunk_offset);
            const std::size_t row_offset = history_size - row * Bins - (Bins - 1U) + chunk_offset;
            const std::size_t chain = RowIlp ? row % chain_count : 0U;
            for (std::size_t hop = 0U; hop < HopCount; ++hop) {
                const std::size_t first_sample = cursors[hop] + row_offset;
                accumulator_re[hop][chain] = vfmaq_f32(
                    accumulator_re[hop][chain], vld1q_f32(history_i + first_sample), coefficient);
                accumulator_im[hop][chain] = vfmaq_f32(
                    accumulator_im[hop][chain], vld1q_f32(history_q + first_sample), coefficient);
            }
        }

        const std::size_t destination_offset = destination_chunk * width;
        const float32x4_t rotation_re = vld1q_f32(rotations_re + destination_offset);
        const float32x4_t rotation_im = vld1q_f32(rotations_im + destination_offset);
        const float32x4_t weight_re = direct ? vld1q_f32(weights_re + destination_offset)
                                             : vdupq_n_f32(0.0f);
        const float32x4_t weight_im = direct ? vld1q_f32(weights_im + destination_offset)
                                             : vdupq_n_f32(0.0f);
        for (std::size_t hop = 0U; hop < HopCount; ++hop) {
            float32x4_t accumulated_re = accumulator_re[hop][0U];
            float32x4_t accumulated_im = accumulator_im[hop][0U];
            if constexpr (RowIlp) {
                accumulated_re = vaddq_f32(
                    vaddq_f32(accumulated_re, accumulator_re[hop][1U]),
                    vaddq_f32(accumulator_re[hop][2U], accumulator_re[hop][3U]));
                accumulated_im = vaddq_f32(
                    vaddq_f32(accumulated_im, accumulator_im[hop][1U]),
                    vaddq_f32(accumulator_im[hop][2U], accumulator_im[hop][3U]));
            }
            const float32x4_t natural_re = reverse_lanes(accumulated_re);
            const float32x4_t natural_im = reverse_lanes(accumulated_im);
            const float32x4_t transformed_re =
                vfmsq_f32(vmulq_f32(natural_re, rotation_re), natural_im, rotation_im);
            const float32x4_t transformed_im =
                vfmaq_f32(vmulq_f32(natural_im, rotation_re), natural_re, rotation_im);
            if (direct) {
                direct_re[hop] = vfmaq_f32(direct_re[hop], transformed_re, weight_re);
                direct_re[hop] = vfmsq_f32(direct_re[hop], transformed_im, weight_im);
                direct_im[hop] = vfmaq_f32(direct_im[hop], transformed_re, weight_im);
                direct_im[hop] = vfmaq_f32(direct_im[hop], transformed_im, weight_re);
            } else {
                vst1q_f32(values_re.data() + hop * Bins + destination_offset, transformed_re);
                vst1q_f32(values_im.data() + hop * Bins + destination_offset, transformed_im);
            }
        }
    }

    if (direct) {
        for (std::size_t hop = 0U; hop < HopCount; ++hop) {
            pfb_store_output(block.outputs[0], output_index + hop,
                             pfb_apply_post_phase(data, 0U, phases[hop],
                                                  horizontal_sum(direct_re[hop]),
                                                  horizontal_sum(direct_im[hop])));
        }
    } else {
        Ifft_neon(values_re.data(), values_im.data(), Bins, HopCount, Bins);
        for (std::size_t hop = 0U; hop < HopCount; ++hop) {
            pfb_emit_transformed_outputs(data, block, output_index + hop, phases[hop],
                                         values_re.data() + hop * Bins,
                                         values_im.data() + hop * Bins);
        }
    }
}

template <std::size_t Bins>
[[nodiscard]] std::size_t process(PfbChannelizerData& data,
                                  const PfbChannelizerBlock& block) noexcept {
    const std::size_t filter_span = PfbChannelizerAccess::rows(data) * Bins;
    const std::size_t history_size = PfbChannelizerAccess::history_size(data);
    const std::size_t batch_limit = std::min<std::size_t>(
        4U, 1U + (history_size - filter_span) / data.decimation());
    return pfb_process_streaming(
        data, block, batch_limit,
        [&](const std::size_t* const cursors, const std::size_t* const phases,
            const std::size_t hop_count, const std::size_t output_index) noexcept {
            if (hop_count == 1U) {
                process_batch<Bins, 1U, true>(data, block, cursors, phases, output_index);
                return;
            }
            switch (hop_count) {
            case 1U:
                process_batch<Bins, 1U>(data, block, cursors, phases, output_index);
                break;
            case 2U:
                process_batch<Bins, 2U>(data, block, cursors, phases, output_index);
                break;
            case 3U:
                process_batch<Bins, 3U>(data, block, cursors, phases, output_index);
                break;
            case 4U:
                process_batch<Bins, 4U>(data, block, cursors, phases, output_index);
                break;
            default:
                break;
            }
        });
}

} // namespace

std::size_t PfbChannelizer_neon(PfbChannelizerData& data,
                                const PfbChannelizerBlock& block) noexcept {
    switch (data.bin_count()) {
    case 4U:
        return process<4U>(data, block);
    case 8U:
        return process<8U>(data, block);
    case 16U:
        return process<16U>(data, block);
    case 32U:
        return process<32U>(data, block);
    default:
        return PfbChannelizer_generic(data, block);
    }
}

} // namespace uni::simd::detail
#else
namespace uni::simd::detail {

std::size_t PfbChannelizer_neon(PfbChannelizerData& data,
                                const PfbChannelizerBlock& block) noexcept {
    return PfbChannelizer_generic(data, block);
}

} // namespace uni::simd::detail
#endif
