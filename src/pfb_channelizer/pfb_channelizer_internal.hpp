#pragma once

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <vector>

#include "common/api_internal.hpp"

namespace uni::simd::detail {

using PfbChannelizerFn = std::size_t (*)(struct PfbChannelizerData&,
                                         const PfbChannelizerBlock&) noexcept;
using PfbChannelizerSupportFn = bool (*)(const struct PfbChannelizerData&) noexcept;

struct PfbChannelizerData final {
    std::size_t bins = 0U;
    std::size_t decimation_value = 0U;
    std::size_t taps = 0U;
    std::size_t rows = 0U;
    std::size_t history_size = 0U;
    std::size_t phase_period = 1U;
    PfbGridOffset offset = PfbGridOffset::integer_bins;
    Backend selected_backend = Backend::generic;
    PfbChannelizerFn process = nullptr;
    std::vector<std::int32_t> selected_bins;
    std::vector<std::size_t> selected_fft_bins;
    std::vector<float> reversed_coefficients;
    std::array<float, pfb_channelizer_max_bins> branch_rotation_re{};
    std::array<float, pfb_channelizer_max_bins> branch_rotation_im{};
    std::vector<float> post_phase_re;
    std::vector<float> post_phase_im;
    std::vector<float> selected_transform_re;
    std::vector<float> selected_transform_im;
    std::vector<float> history;
    std::size_t cursor = 0U;
    std::size_t decimation_phase = 0U;
    std::size_t post_phase = 0U;

    [[nodiscard]] std::size_t bin_count() const noexcept { return bins; }
    [[nodiscard]] std::size_t decimation() const noexcept { return decimation_value; }
    [[nodiscard]] std::size_t tap_count() const noexcept { return taps; }
    [[nodiscard]] std::size_t selected_output_count() const noexcept { return selected_bins.size(); }
    [[nodiscard]] PfbGridOffset grid_offset() const noexcept { return offset; }
    [[nodiscard]] std::span<const std::int32_t> logical_bins() const noexcept { return selected_bins; }
};

struct PfbChannelizerAccess final {
    [[nodiscard]] static std::size_t rows(const PfbChannelizerData& data) noexcept { return data.rows; }
    [[nodiscard]] static std::size_t history_size(const PfbChannelizerData& data) noexcept { return data.history_size; }
    [[nodiscard]] static std::size_t phase_period(const PfbChannelizerData& data) noexcept { return data.phase_period; }
    [[nodiscard]] static const float* reversed_coefficients(const PfbChannelizerData& data) noexcept {
        return data.reversed_coefficients.data();
    }
    [[nodiscard]] static const float* branch_rotation_re(const PfbChannelizerData& data) noexcept {
        return data.branch_rotation_re.data();
    }
    [[nodiscard]] static const float* branch_rotation_im(const PfbChannelizerData& data) noexcept {
        return data.branch_rotation_im.data();
    }
    [[nodiscard]] static const float* post_phase_re(const PfbChannelizerData& data) noexcept {
        return data.post_phase_re.data();
    }
    [[nodiscard]] static const float* post_phase_im(const PfbChannelizerData& data) noexcept {
        return data.post_phase_im.data();
    }
    [[nodiscard]] static const float* selected_transform_re(const PfbChannelizerData& data) noexcept {
        return data.selected_transform_re.data();
    }
    [[nodiscard]] static const float* selected_transform_im(const PfbChannelizerData& data) noexcept {
        return data.selected_transform_im.data();
    }
    [[nodiscard]] static const std::size_t* selected_fft_bins(const PfbChannelizerData& data) noexcept {
        return data.selected_fft_bins.data();
    }
    [[nodiscard]] static std::size_t cursor(const PfbChannelizerData& data) noexcept { return data.cursor; }
    [[nodiscard]] static std::size_t decimation_phase(const PfbChannelizerData& data) noexcept {
        return data.decimation_phase;
    }
    [[nodiscard]] static std::size_t post_phase(const PfbChannelizerData& data) noexcept { return data.post_phase; }
    [[nodiscard]] static float* history_i(PfbChannelizerData& data) noexcept { return data.history.data(); }
    [[nodiscard]] static const float* history_i(const PfbChannelizerData& data) noexcept {
        return data.history.data();
    }
    [[nodiscard]] static float* history_q(PfbChannelizerData& data) noexcept {
        return data.history.data() + 2U * data.history_size;
    }
    [[nodiscard]] static const float* history_q(const PfbChannelizerData& data) noexcept {
        return data.history.data() + 2U * data.history_size;
    }
    static void set_cursor(PfbChannelizerData& data, const std::size_t value) noexcept { data.cursor = value; }
    static void set_decimation_phase(PfbChannelizerData& data, const std::size_t value) noexcept {
        data.decimation_phase = value;
    }
    static void set_post_phase(PfbChannelizerData& data, const std::size_t value) noexcept { data.post_phase = value; }
};

[[nodiscard]] std::expected<std::unique_ptr<PfbChannelizerData>, Result>
make_pfb_channelizer_data(const PfbChannelizerConfig& config, PfbChannelizerFn candidate,
                          PfbChannelizerSupportFn supports, Backend backend) noexcept;

[[nodiscard]] std::size_t PfbChannelizer_generic(PfbChannelizerData& data,
                                                 const PfbChannelizerBlock& block) noexcept;
[[nodiscard]] std::size_t PfbChannelizer_avx2fma(PfbChannelizerData& data,
                                                 const PfbChannelizerBlock& block) noexcept;
[[nodiscard]] std::size_t PfbChannelizer_neon(PfbChannelizerData& data,
                                               const PfbChannelizerBlock& block) noexcept;
[[nodiscard]] bool PfbChannelizer_supports_all(const PfbChannelizerData&) noexcept;

[[nodiscard]] inline std::size_t pfb_output_count_unchecked(const PfbChannelizerData& data,
                                                             const std::size_t input_count) noexcept {
    if (input_count == 0U) {
        return 0U;
    }
    const std::size_t first = data.decimation_phase == 0U ? 0U : data.decimation_value - data.decimation_phase;
    return first >= input_count ? 0U : 1U + (input_count - 1U - first) / data.decimation_value;
}

[[nodiscard]] inline std::complex<float> pfb_apply_post_phase(const PfbChannelizerData& data,
                                                               const std::size_t output,
                                                               const std::size_t phase,
                                                               const float real,
                                                               const float imag) noexcept {
    const std::size_t table_index = output * data.phase_period + phase;
    const float phase_re = data.post_phase_re[table_index];
    const float phase_im = data.post_phase_im[table_index];
    if (phase_im == 0.0f && (phase_re == 1.0f || phase_re == -1.0f)) {
        return {phase_re * real, phase_re * imag};
    }
    if (phase_re == 0.0f && (phase_im == 1.0f || phase_im == -1.0f)) {
        return {-phase_im * imag, phase_im * real};
    }
    return {real * phase_re - imag * phase_im, real * phase_im + imag * phase_re};
}

inline void pfb_store_output(const std::span<float> output, const std::size_t index,
                             const std::complex<float> value) noexcept {
    output[2U * index] = value.real();
    output[2U * index + 1U] = value.imag();
}

inline void pfb_emit_direct_output(const PfbChannelizerData& data,
                                   const PfbChannelizerBlock& block,
                                   const std::size_t output_index, const std::size_t phase,
                                   const float* const transform_re,
                                   const float* const transform_im) noexcept {
    const std::size_t bins = data.bin_count();
    const float* weights_re = PfbChannelizerAccess::selected_transform_re(data);
    const float* weights_im = PfbChannelizerAccess::selected_transform_im(data);
    float value_re = 0.0f;
    float value_im = 0.0f;
    for (std::size_t branch = 0U; branch < bins; ++branch) {
        value_re += transform_re[branch] * weights_re[branch] -
                    transform_im[branch] * weights_im[branch];
        value_im += transform_re[branch] * weights_im[branch] +
                    transform_im[branch] * weights_re[branch];
    }
    pfb_store_output(block.outputs[0], output_index,
                     pfb_apply_post_phase(data, 0U, phase, value_re, value_im));
}

inline void pfb_emit_transformed_outputs(const PfbChannelizerData& data,
                                         const PfbChannelizerBlock& block,
                                         const std::size_t output_index,
                                         const std::size_t phase,
                                         const float* const transform_re,
                                         const float* const transform_im) noexcept {
    const std::size_t selected_count = data.selected_output_count();
    const std::size_t* output_bins = PfbChannelizerAccess::selected_fft_bins(data);
    for (std::size_t output = 0U; output < selected_count; ++output) {
        const std::size_t bin = output_bins[output];
        pfb_store_output(block.outputs[output], output_index,
                         pfb_apply_post_phase(data, output, phase,
                                              transform_re[bin], transform_im[bin]));
    }
}

template <typename ProcessBatch>
[[nodiscard]] inline std::size_t pfb_process_streaming(PfbChannelizerData& data,
                                                        const PfbChannelizerBlock& block,
                                                        const std::size_t batch_limit,
                                                        ProcessBatch&& process_batch) noexcept {
    const std::size_t history_size = PfbChannelizerAccess::history_size(data);
    const std::size_t history_mask = history_size - 1U;
    const std::size_t decimation = data.decimation();
    const std::size_t phase_period = PfbChannelizerAccess::phase_period(data);
    const std::size_t selected_count = data.selected_output_count();
    float* history_i = PfbChannelizerAccess::history_i(data);
    float* history_q = PfbChannelizerAccess::history_q(data);
    std::size_t cursor = PfbChannelizerAccess::cursor(data);
    std::size_t decimation_phase = PfbChannelizerAccess::decimation_phase(data);
    std::size_t post_phase = PfbChannelizerAccess::post_phase(data);
    std::size_t produced = 0U;
    std::size_t queued = 0U;
    std::array<std::size_t, 4U> queued_cursors{};
    std::array<std::size_t, 4U> queued_phases{};

    for (std::size_t input_index = 0U; input_index < block.input.size() / 2U; ++input_index) {
        const float sample_re = block.input[2U * input_index];
        const float sample_im = block.input[2U * input_index + 1U];
        history_i[cursor] = sample_re;
        history_i[cursor + history_size] = sample_re;
        history_q[cursor] = sample_im;
        history_q[cursor + history_size] = sample_im;

        if (decimation_phase == 0U) {
            if (selected_count == 0U) {
                ++produced;
            } else {
                queued_cursors[queued] = cursor;
                queued_phases[queued] = post_phase;
                ++queued;
                if (queued == batch_limit) {
                    process_batch(queued_cursors.data(), queued_phases.data(), queued, produced);
                    produced += queued;
                    queued = 0U;
                }
            }
            post_phase = post_phase + 1U == phase_period ? 0U : post_phase + 1U;
        }

        cursor = (cursor + 1U) & history_mask;
        decimation_phase = decimation_phase + 1U == decimation ? 0U : decimation_phase + 1U;
    }

    if (queued != 0U) {
        process_batch(queued_cursors.data(), queued_phases.data(), queued, produced);
        produced += queued;
    }
    PfbChannelizerAccess::set_cursor(data, cursor);
    PfbChannelizerAccess::set_decimation_phase(data, decimation_phase);
    PfbChannelizerAccess::set_post_phase(data, post_phase);
    return produced;
}

} // namespace uni::simd::detail
