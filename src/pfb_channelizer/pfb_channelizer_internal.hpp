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
    std::vector<float> reversed_coefficients;
    std::array<float, pfb_channelizer_max_bins> branch_rotation_re{};
    std::array<float, pfb_channelizer_max_bins> branch_rotation_im{};
    std::vector<float> post_phase_re;
    std::vector<float> post_phase_im;
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
    [[nodiscard]] static std::size_t cursor(const PfbChannelizerData& data) noexcept { return data.cursor; }
    [[nodiscard]] static std::size_t decimation_phase(const PfbChannelizerData& data) noexcept {
        return data.decimation_phase;
    }
    [[nodiscard]] static std::size_t post_phase(const PfbChannelizerData& data) noexcept { return data.post_phase; }
    [[nodiscard]] static float* history_i(PfbChannelizerData& data) noexcept { return data.history.data(); }
    [[nodiscard]] static float* history_q(PfbChannelizerData& data) noexcept {
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
[[nodiscard]] bool PfbChannelizer_supports_m8_d4(const PfbChannelizerData& data) noexcept;

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

} // namespace uni::simd::detail
