#include "pfb_channelizer/pfb_channelizer_internal.hpp"
#include "ifft_cf32/ifft_cf32_internal.hpp"

#include <array>
#include <cstddef>

#include <immintrin.h>

namespace uni::simd::detail {
namespace {

struct ScalarComplex {
    float real;
    float imag;
};

[[nodiscard]] inline __m256 reverse_lanes(const __m256 value) noexcept {
    const __m256i reverse = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    return _mm256_permutevar8x32_ps(value, reverse);
}

[[nodiscard]] inline ScalarComplex multiply_i(const ScalarComplex value, const float sign) noexcept {
    return {-sign * value.imag, sign * value.real};
}

inline void store_complex(float* const output, const std::size_t index,
                          const ScalarComplex value) noexcept {
    output[2U * index] = value.real;
    output[2U * index + 1U] = value.imag;
}

inline void emit_target_values(const ScalarComplex value0, const ScalarComplex value1,
                               const ScalarComplex value2, const ScalarComplex value3,
                               float* const output0, float* const output1,
                               float* const output2, float* const output3,
                               const std::size_t output_index, const std::size_t phase) noexcept {
    if (phase == 0U) {
        store_complex(output0, output_index, value0);
        store_complex(output1, output_index, value1);
        store_complex(output2, output_index, value2);
        store_complex(output3, output_index, value3);
    } else if (phase == 1U) {
        store_complex(output0, output_index, multiply_i(value0, -1.0f));
        store_complex(output1, output_index, multiply_i(value1, 1.0f));
        store_complex(output2, output_index, multiply_i(value2, -1.0f));
        store_complex(output3, output_index, multiply_i(value3, 1.0f));
    } else if (phase == 2U) {
        store_complex(output0, output_index, {-value0.real, -value0.imag});
        store_complex(output1, output_index, {-value1.real, -value1.imag});
        store_complex(output2, output_index, {-value2.real, -value2.imag});
        store_complex(output3, output_index, {-value3.real, -value3.imag});
    } else {
        store_complex(output0, output_index, multiply_i(value0, 1.0f));
        store_complex(output1, output_index, multiply_i(value1, -1.0f));
        store_complex(output2, output_index, multiply_i(value2, 1.0f));
        store_complex(output3, output_index, multiply_i(value3, -1.0f));
    }
}

[[nodiscard]] inline bool is_target_four(const PfbChannelizerData& plan) noexcept {
    const auto logical_bins = plan.logical_bins();
    return plan.tap_count() == 169U && plan.selected_output_count() == 4U &&
           plan.grid_offset() == PfbGridOffset::half_bins && logical_bins[0] == -2 &&
           logical_bins[1] == -1 && logical_bins[2] == 0 && logical_bins[3] == 1;
}

inline void emit_target_hop(const __m256 accumulated_reversed_re, const __m256 accumulated_reversed_im,
                            const __m256 rotation_re, const __m256 rotation_im,
                            float* output0, float* output1, float* output2, float* output3,
                            const std::size_t output_index, const std::size_t phase) noexcept {
    const __m256 accumulator_re = reverse_lanes(accumulated_reversed_re);
    const __m256 accumulator_im = reverse_lanes(accumulated_reversed_im);
    const __m256 rotated_re =
        _mm256_fmsub_ps(accumulator_re, rotation_re, _mm256_mul_ps(accumulator_im, rotation_im));
    const __m256 rotated_im =
        _mm256_fmadd_ps(accumulator_re, rotation_im, _mm256_mul_ps(accumulator_im, rotation_re));
    alignas(32) std::array<float, 8U> fft_re{};
    alignas(32) std::array<float, 8U> fft_im{};
    _mm256_store_ps(fft_re.data(), rotated_re);
    _mm256_store_ps(fft_im.data(), rotated_im);
    Ifft8_avx2_fma(fft_re.data(), fft_im.data(), 8U);

    const ScalarComplex value0{fft_re[6], fft_im[6]};
    const ScalarComplex value1{fft_re[7], fft_im[7]};
    const ScalarComplex value2{fft_re[0], fft_im[0]};
    const ScalarComplex value3{fft_re[1], fft_im[1]};
    emit_target_values(value0, value1, value2, value3, output0, output1,
                       output2, output3, output_index, phase);
}

inline void process_target_hop(const float* coefficients, const float* history_i, const float* history_q,
                               const std::size_t history_size, const std::size_t cursor,
                               const __m256 rotation_re, const __m256 rotation_im,
                               float* output0, float* output1, float* output2, float* output3,
                               const std::size_t output_index, const std::size_t phase) noexcept {
    __m256 accumulator_re = _mm256_setzero_ps();
    __m256 accumulator_im = _mm256_setzero_ps();
    for (std::size_t row = 0U; row < 22U; ++row) {
        const __m256 coefficient = _mm256_loadu_ps(coefficients + row * 8U);
        const std::size_t first_sample = cursor + history_size - row * 8U - 7U;
        accumulator_re = _mm256_fmadd_ps(_mm256_loadu_ps(history_i + first_sample), coefficient,
                                         accumulator_re);
        accumulator_im = _mm256_fmadd_ps(_mm256_loadu_ps(history_q + first_sample), coefficient,
                                         accumulator_im);
    }
    emit_target_hop(accumulator_re, accumulator_im, rotation_re, rotation_im, output0, output1,
                    output2, output3, output_index, phase);
}

inline void process_four_target_hops(const float* coefficients, const float* history_i,
                                     const float* history_q, const std::size_t history_size,
                                     const std::array<std::size_t, 4U>& cursors,
                                     const __m256 rotation_re, const __m256 rotation_im,
                                      float* output0, float* output1, float* output2, float* output3,
                                     const std::size_t output_index, const std::size_t first_phase) noexcept {
    __m256 accumulator_re0 = _mm256_setzero_ps();
    __m256 accumulator_re1 = _mm256_setzero_ps();
    __m256 accumulator_re2 = _mm256_setzero_ps();
    __m256 accumulator_re3 = _mm256_setzero_ps();
    __m256 accumulator_im0 = _mm256_setzero_ps();
    __m256 accumulator_im1 = _mm256_setzero_ps();
    __m256 accumulator_im2 = _mm256_setzero_ps();
    __m256 accumulator_im3 = _mm256_setzero_ps();
    for (std::size_t row = 0U; row < 22U; ++row) {
        const __m256 coefficient = _mm256_loadu_ps(coefficients + row * 8U);
        const std::size_t row_offset = history_size - row * 8U - 7U;
        accumulator_re0 = _mm256_fmadd_ps(_mm256_loadu_ps(history_i + cursors[0] + row_offset),
                                          coefficient, accumulator_re0);
        accumulator_im0 = _mm256_fmadd_ps(_mm256_loadu_ps(history_q + cursors[0] + row_offset),
                                          coefficient, accumulator_im0);
        accumulator_re1 = _mm256_fmadd_ps(_mm256_loadu_ps(history_i + cursors[1] + row_offset),
                                          coefficient, accumulator_re1);
        accumulator_im1 = _mm256_fmadd_ps(_mm256_loadu_ps(history_q + cursors[1] + row_offset),
                                          coefficient, accumulator_im1);
        accumulator_re2 = _mm256_fmadd_ps(_mm256_loadu_ps(history_i + cursors[2] + row_offset),
                                          coefficient, accumulator_re2);
        accumulator_im2 = _mm256_fmadd_ps(_mm256_loadu_ps(history_q + cursors[2] + row_offset),
                                          coefficient, accumulator_im2);
        accumulator_re3 = _mm256_fmadd_ps(_mm256_loadu_ps(history_i + cursors[3] + row_offset),
                                          coefficient, accumulator_re3);
        accumulator_im3 = _mm256_fmadd_ps(_mm256_loadu_ps(history_q + cursors[3] + row_offset),
                                          coefficient, accumulator_im3);
    }

    emit_target_hop(accumulator_re0, accumulator_im0, rotation_re, rotation_im, output0, output1,
                    output2, output3, output_index + 0U, (first_phase + 0U) & 3U);
    emit_target_hop(accumulator_re1, accumulator_im1, rotation_re, rotation_im, output0, output1,
                    output2, output3, output_index + 1U, (first_phase + 1U) & 3U);
    emit_target_hop(accumulator_re2, accumulator_im2, rotation_re, rotation_im, output0, output1,
                    output2, output3, output_index + 2U, (first_phase + 2U) & 3U);
    emit_target_hop(accumulator_re3, accumulator_im3, rotation_re, rotation_im, output0, output1,
                    output2, output3, output_index + 3U, (first_phase + 3U) & 3U);
}

[[nodiscard]] std::size_t process_target_four(PfbChannelizerData& data,
                                               const PfbChannelizerBlock& block) noexcept {
    const auto& plan = data;
    auto& state = data;
    constexpr std::size_t decimation = 4U;
    const std::size_t history_size = PfbChannelizerAccess::history_size(plan);
    const std::size_t history_mask = history_size - 1U;
    const float* coefficients = PfbChannelizerAccess::reversed_coefficients(plan);
    const __m256 rotation_re = _mm256_loadu_ps(PfbChannelizerAccess::branch_rotation_re(plan));
    const __m256 rotation_im = _mm256_loadu_ps(PfbChannelizerAccess::branch_rotation_im(plan));
    float* history_i = PfbChannelizerAccess::history_i(state);
    float* history_q = PfbChannelizerAccess::history_q(state);
    std::size_t cursor = PfbChannelizerAccess::cursor(state);
    std::size_t decimation_phase = PfbChannelizerAccess::decimation_phase(state);
    std::size_t post_phase = PfbChannelizerAccess::post_phase(state);
    std::size_t produced = 0U;
    std::size_t queued = 0U;
    std::size_t first_queued_phase = 0U;
    std::array<std::size_t, 4U> queued_cursors{};
    auto* output0 = block.outputs[0].data();
    auto* output1 = block.outputs[1].data();
    auto* output2 = block.outputs[2].data();
    auto* output3 = block.outputs[3].data();

    for (std::size_t input_index = 0U; input_index < block.input.size() / 2U; ++input_index) {
        const float sample_re = block.input[2U * input_index];
        const float sample_im = block.input[2U * input_index + 1U];
        history_i[cursor] = sample_re;
        history_i[cursor + history_size] = sample_re;
        history_q[cursor] = sample_im;
        history_q[cursor + history_size] = sample_im;

        if (decimation_phase == 0U) {
            if (queued == 0U) {
                first_queued_phase = post_phase;
            }
            queued_cursors[queued++] = cursor;
            post_phase = (post_phase + 1U) & 3U;
            if (queued == 4U) {
                process_four_target_hops(coefficients, history_i, history_q, history_size,
                                         queued_cursors, rotation_re, rotation_im, output0, output1,
                                         output2, output3, produced, first_queued_phase);
                produced += 4U;
                queued = 0U;
            }
        }

        cursor = (cursor + 1U) & history_mask;
        decimation_phase = decimation_phase + 1U == decimation ? 0U : decimation_phase + 1U;
    }

    for (std::size_t index = 0U; index < queued; ++index) {
        process_target_hop(coefficients, history_i, history_q, history_size, queued_cursors[index],
                           rotation_re, rotation_im, output0, output1, output2, output3,
                           produced + index, (first_queued_phase + index) & 3U);
    }
    produced += queued;
    PfbChannelizerAccess::set_cursor(state, cursor);
    PfbChannelizerAccess::set_decimation_phase(state, decimation_phase);
    PfbChannelizerAccess::set_post_phase(state, post_phase);
    return produced;
}

} // namespace

std::size_t PfbChannelizer_avx2fma(PfbChannelizerData& data, const PfbChannelizerBlock& block) noexcept {
    const auto& plan = data;
    auto& state = data;
    if (is_target_four(plan) && PfbChannelizerAccess::rows(plan) == 22U &&
        PfbChannelizerAccess::history_size(plan) >= 188U) {
        return process_target_four(data, block);
    }

    constexpr std::size_t bins = 8U;
    constexpr std::size_t decimation = 4U;
    const std::size_t rows = PfbChannelizerAccess::rows(plan);
    const std::size_t history_size = PfbChannelizerAccess::history_size(plan);
    const std::size_t history_mask = history_size - 1U;
    const std::size_t selected_count = plan.selected_output_count();
    const std::size_t phase_period = PfbChannelizerAccess::phase_period(plan);
    const float* coefficients = PfbChannelizerAccess::reversed_coefficients(plan);
    const __m256 rotation_re = _mm256_loadu_ps(PfbChannelizerAccess::branch_rotation_re(plan));
    const __m256 rotation_im = _mm256_loadu_ps(PfbChannelizerAccess::branch_rotation_im(plan));
    float* history_i = PfbChannelizerAccess::history_i(state);
    float* history_q = PfbChannelizerAccess::history_q(state);
    std::size_t cursor = PfbChannelizerAccess::cursor(state);
    std::size_t decimation_phase = PfbChannelizerAccess::decimation_phase(state);
    std::size_t post_phase = PfbChannelizerAccess::post_phase(state);
    std::size_t produced = 0U;
    std::array<std::size_t, pfb_channelizer_max_outputs> output_bins{};
    const auto logical_bins = plan.logical_bins();
    for (std::size_t output = 0U; output < selected_count; ++output) {
        output_bins[output] = static_cast<std::size_t>(logical_bins[output] < 0 ? logical_bins[output] + 8
                                                                              : logical_bins[output]);
    }
    const bool target_four = selected_count == 4U && plan.grid_offset() == PfbGridOffset::half_bins &&
                             logical_bins[0] == -2 && logical_bins[1] == -1 &&
                             logical_bins[2] == 0 && logical_bins[3] == 1;
    auto* target_output0 = block.outputs[0].data();
    auto* target_output1 = block.outputs[1].data();
    auto* target_output2 = block.outputs[2].data();
    auto* target_output3 = block.outputs[3].data();

    alignas(32) std::array<float, pfb_channelizer_max_bins> fft_re{};
    alignas(32) std::array<float, pfb_channelizer_max_bins> fft_im{};

    for (std::size_t input_index = 0U; input_index < block.input.size() / 2U; ++input_index) {
        const float sample_re = block.input[2U * input_index];
        const float sample_im = block.input[2U * input_index + 1U];
        history_i[cursor] = sample_re;
        history_i[cursor + history_size] = sample_re;
        history_q[cursor] = sample_im;
        history_q[cursor + history_size] = sample_im;

        if (decimation_phase == 0U) {
            if (selected_count != 0U) {
                __m256 accumulator_re0 = _mm256_setzero_ps();
                __m256 accumulator_re1 = _mm256_setzero_ps();
                __m256 accumulator_re2 = _mm256_setzero_ps();
                __m256 accumulator_re3 = _mm256_setzero_ps();
                __m256 accumulator_im0 = _mm256_setzero_ps();
                __m256 accumulator_im1 = _mm256_setzero_ps();
                __m256 accumulator_im2 = _mm256_setzero_ps();
                __m256 accumulator_im3 = _mm256_setzero_ps();
                const auto accumulate_row = [&](const std::size_t row, __m256& accumulator_re,
                                                __m256& accumulator_im) noexcept {
                    const __m256 coefficient = _mm256_loadu_ps(coefficients + row * bins);
                    const std::size_t first_sample = cursor + history_size - row * bins - (bins - 1U);
                    const __m256 samples_re = _mm256_loadu_ps(history_i + first_sample);
                    const __m256 samples_im = _mm256_loadu_ps(history_q + first_sample);
                    accumulator_re = _mm256_fmadd_ps(samples_re, coefficient, accumulator_re);
                    accumulator_im = _mm256_fmadd_ps(samples_im, coefficient, accumulator_im);
                };
                std::size_t row = 0U;
                for (; row + 4U <= rows; row += 4U) {
                    accumulate_row(row + 0U, accumulator_re0, accumulator_im0);
                    accumulate_row(row + 1U, accumulator_re1, accumulator_im1);
                    accumulate_row(row + 2U, accumulator_re2, accumulator_im2);
                    accumulate_row(row + 3U, accumulator_re3, accumulator_im3);
                }
                for (; row < rows; ++row) {
                    accumulate_row(row, accumulator_re0, accumulator_im0);
                }

                const __m256 accumulator_re = reverse_lanes(
                    _mm256_add_ps(_mm256_add_ps(accumulator_re0, accumulator_re1),
                                  _mm256_add_ps(accumulator_re2, accumulator_re3)));
                const __m256 accumulator_im = reverse_lanes(
                    _mm256_add_ps(_mm256_add_ps(accumulator_im0, accumulator_im1),
                                  _mm256_add_ps(accumulator_im2, accumulator_im3)));

                const __m256 rotated_re =
                    _mm256_fmsub_ps(accumulator_re, rotation_re, _mm256_mul_ps(accumulator_im, rotation_im));
                const __m256 rotated_im =
                    _mm256_fmadd_ps(accumulator_re, rotation_im, _mm256_mul_ps(accumulator_im, rotation_re));
                _mm256_store_ps(fft_re.data(), rotated_re);
                _mm256_store_ps(fft_im.data(), rotated_im);

                Ifft8_avx2_fma(fft_re.data(), fft_im.data(), bins);
                if (target_four) {
                    const ScalarComplex value0{fft_re[6], fft_im[6]};
                    const ScalarComplex value1{fft_re[7], fft_im[7]};
                    const ScalarComplex value2{fft_re[0], fft_im[0]};
                    const ScalarComplex value3{fft_re[1], fft_im[1]};
                    emit_target_values(value0, value1, value2, value3, target_output0,
                                       target_output1, target_output2, target_output3,
                                       produced, post_phase);
                } else {
                    for (std::size_t output = 0U; output < selected_count; ++output) {
                        const std::size_t fft_bin = output_bins[output];
                        pfb_store_output(block.outputs[output], produced,
                                         pfb_apply_post_phase(plan, output, post_phase,
                                                              fft_re[fft_bin], fft_im[fft_bin]));
                    }
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
