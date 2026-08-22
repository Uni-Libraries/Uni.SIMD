#include "pfb_channelizer/pfb_channelizer_internal.hpp"
#include "ifft_cf32/ifft_cf32_internal.hpp"

#include <array>
#include <cstddef>

namespace uni::simd::detail {

std::size_t PfbChannelizer_generic(PfbChannelizerData& data, const PfbChannelizerBlock& block) noexcept {
    const auto& plan = data;
    auto& state = data;
    const std::size_t bins = plan.bin_count();
    const std::size_t decimation = plan.decimation();
    const std::size_t rows = PfbChannelizerAccess::rows(plan);
    const std::size_t history_size = PfbChannelizerAccess::history_size(plan);
    const std::size_t history_mask = history_size - 1U;
    const std::size_t selected_count = plan.selected_output_count();
    const std::size_t phase_period = PfbChannelizerAccess::phase_period(plan);
    const float* coefficients = PfbChannelizerAccess::reversed_coefficients(plan);
    const float* rotation_re = PfbChannelizerAccess::branch_rotation_re(plan);
    const float* rotation_im = PfbChannelizerAccess::branch_rotation_im(plan);
    float* history_i = PfbChannelizerAccess::history_i(state);
    float* history_q = PfbChannelizerAccess::history_q(state);
    std::size_t cursor = PfbChannelizerAccess::cursor(state);
    std::size_t decimation_phase = PfbChannelizerAccess::decimation_phase(state);
    std::size_t post_phase = PfbChannelizerAccess::post_phase(state);
    std::size_t produced = 0U;

    alignas(64) std::array<float, pfb_channelizer_max_bins> fft_re{};
    alignas(64) std::array<float, pfb_channelizer_max_bins> fft_im{};

    for (const auto sample : block.input) {
        const float sample_re = sample.real();
        const float sample_im = sample.imag();
        history_i[cursor] = sample_re;
        history_i[cursor + history_size] = sample_re;
        history_q[cursor] = sample_im;
        history_q[cursor + history_size] = sample_im;

        if (decimation_phase == 0U) {
            if (selected_count != 0U) {
                std::fill_n(fft_re.data(), bins, 0.0f);
                std::fill_n(fft_im.data(), bins, 0.0f);
                for (std::size_t row = 0U; row < rows; ++row) {
                    const std::size_t first_sample = cursor + history_size - row * bins - (bins - 1U);
                    const float* row_coefficients = coefficients + row * bins;
                    for (std::size_t lane = 0U; lane < bins; ++lane) {
                        const std::size_t branch = bins - 1U - lane;
                        fft_re[branch] += history_i[first_sample + lane] * row_coefficients[lane];
                        fft_im[branch] += history_q[first_sample + lane] * row_coefficients[lane];
                    }
                }
                for (std::size_t branch = 0U; branch < bins; ++branch) {
                    const float accumulator_re = fft_re[branch];
                    const float accumulator_im = fft_im[branch];
                    fft_re[branch] = accumulator_re * rotation_re[branch] - accumulator_im * rotation_im[branch];
                    fft_im[branch] = accumulator_re * rotation_im[branch] + accumulator_im * rotation_re[branch];
                }

                Ifft_generic(fft_re.data(), fft_im.data(), bins);
                const auto logical_bins = plan.logical_bins();
                for (std::size_t output = 0U; output < selected_count; ++output) {
                    const std::int32_t logical_bin = logical_bins[output];
                    const std::size_t fft_bin = static_cast<std::size_t>(
                        logical_bin < 0 ? logical_bin + static_cast<std::int32_t>(bins) : logical_bin);
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
