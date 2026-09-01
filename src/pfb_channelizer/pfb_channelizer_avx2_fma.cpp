#include "ifft_cf32/ifft_cf32_internal.hpp"
#include "pfb_channelizer/pfb_channelizer_internal.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include <immintrin.h>

namespace uni::simd::detail {
namespace {

[[nodiscard]] inline __m128 reverse_lanes(const __m128 value) noexcept { return _mm_shuffle_ps(value, value, _MM_SHUFFLE(0, 1, 2, 3)); }

[[nodiscard]] inline __m256 reverse_lanes(const __m256 value) noexcept {
    const __m256i reverse = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    return _mm256_permutevar8x32_ps(value, reverse);
}

[[nodiscard]] inline float horizontal_sum(const __m128 value) noexcept {
    const __m128 pairs = _mm_hadd_ps(value, value);
    return _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
}

[[nodiscard]] inline float horizontal_sum(const __m256 value) noexcept {
    return horizontal_sum(_mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1)));
}

[[nodiscard]] inline __m256 pack_complex_hops(const __m128 real, const __m128 imag) noexcept {
    return _mm256_insertf128_ps(_mm256_castps128_ps256(real), imag, 1);
}

inline void transpose_four_hops(const __m256* const input, __m256* const pairs) noexcept {
    const __m256 t0 = _mm256_unpacklo_ps(input[0U], input[1U]);
    const __m256 t1 = _mm256_unpackhi_ps(input[0U], input[1U]);
    const __m256 t2 = _mm256_unpacklo_ps(input[2U], input[3U]);
    const __m256 t3 = _mm256_unpackhi_ps(input[2U], input[3U]);
    pairs[0U] = _mm256_shuffle_ps(t0, t2, 0x44);
    pairs[1U] = _mm256_shuffle_ps(t0, t2, 0xEE);
    pairs[2U] = _mm256_shuffle_ps(t1, t3, 0x44);
    pairs[3U] = _mm256_shuffle_ps(t1, t3, 0xEE);
}

[[nodiscard]] inline __m256 multiply_complex(const __m256 value, const float real, const float imag) noexcept {
    const __m256 swapped = _mm256_permute2f128_ps(value, value, 0x01);
    const __m256 imag_sign = _mm256_setr_ps(-imag, -imag, -imag, -imag, imag, imag, imag, imag);
    return _mm256_fmadd_ps(swapped, imag_sign, _mm256_mul_ps(value, _mm256_set1_ps(real)));
}

[[nodiscard]] inline __m256 multiply_by_i(const __m256 value) noexcept {
    const __m256 sign = _mm256_castsi256_ps(_mm256_setr_epi32(-2147483647 - 1, -2147483647 - 1, -2147483647 - 1, -2147483647 - 1, 0, 0, 0, 0));
    return _mm256_xor_ps(_mm256_permute2f128_ps(value, value, 0x01), sign);
}

inline void butterfly(__m256& even, __m256& odd) noexcept {
    const __m256 left = even;
    const __m256 right = odd;
    even = _mm256_add_ps(left, right);
    odd = _mm256_sub_ps(left, right);
}

struct alignas(32) D4x4PhaseMap final {
    std::array<std::int32_t, 8U> lanes;
    std::array<float, 8U> signs_a;
    std::array<float, 8U> signs_b;
};

constexpr float negative_zero = -0.0f;
alignas(32) constexpr std::array<D4x4PhaseMap, 4U> d4x4_phase_maps{{
    {{0, 4, 5, 1, 2, 6, 7, 3},
     {0.0f, 0.0f, 0.0f, negative_zero, negative_zero, negative_zero, negative_zero, 0.0f},
     {0.0f, 0.0f, negative_zero, 0.0f, negative_zero, negative_zero, 0.0f, negative_zero}},
    {{4, 0, 1, 5, 6, 2, 3, 7},
     {0.0f, negative_zero, negative_zero, negative_zero, negative_zero, 0.0f, 0.0f, 0.0f},
     {negative_zero, 0.0f, negative_zero, negative_zero, 0.0f, negative_zero, 0.0f, 0.0f}},
    {{0, 4, 5, 1, 2, 6, 7, 3},
     {negative_zero, negative_zero, negative_zero, 0.0f, 0.0f, 0.0f, 0.0f, negative_zero},
     {negative_zero, negative_zero, 0.0f, negative_zero, 0.0f, 0.0f, negative_zero, 0.0f}},
    {{4, 0, 1, 5, 6, 2, 3, 7},
     {negative_zero, 0.0f, 0.0f, 0.0f, 0.0f, negative_zero, negative_zero, negative_zero},
     {0.0f, negative_zero, 0.0f, 0.0f, negative_zero, 0.0f, negative_zero, negative_zero}},
}};

static_assert(alignof(D4x4PhaseMap) == 32U);
static_assert(sizeof(D4x4PhaseMap) == 96U);

template <bool ExactD4x4, bool AlignedD4x4>
inline void ifft8_four_hops_emit(const PfbChannelizerData& data, const PfbChannelizerBlock& block, const std::size_t output_index,
                                 const std::size_t* const phases, const __m256* const input_re, const __m256* const input_im) noexcept {
    // Each YMM becomes one complex bin across four hops: real in the low half,
    // imaginary in the high half. DIF leaves output bins in bit-reversed slots.
    __m256 real_pairs[4U];
    __m256 imag_pairs[4U];
    transpose_four_hops(input_re, real_pairs);
    transpose_four_hops(input_im, imag_pairs);

    __m256 bins[8U];
    for (std::size_t index = 0U; index < 4U; ++index) {
        bins[index] = pack_complex_hops(_mm256_castps256_ps128(real_pairs[index]), _mm256_castps256_ps128(imag_pairs[index]));
        bins[index + 4U] = pack_complex_hops(_mm256_extractf128_ps(real_pairs[index], 1), _mm256_extractf128_ps(imag_pairs[index], 1));
    }

    constexpr float root_half = 0.70710678118654752440f;
    for (std::size_t index = 0U; index < 4U; ++index) {
        butterfly(bins[index], bins[index + 4U]);
    }
    bins[5U] = multiply_complex(bins[5U], root_half, root_half);
    bins[6U] = multiply_by_i(bins[6U]);
    bins[7U] = multiply_complex(bins[7U], -root_half, root_half);

    butterfly(bins[0U], bins[2U]);
    butterfly(bins[1U], bins[3U]);
    bins[3U] = multiply_by_i(bins[3U]);
    butterfly(bins[4U], bins[6U]);
    butterfly(bins[5U], bins[7U]);
    bins[7U] = multiply_by_i(bins[7U]);

    if constexpr (ExactD4x4) {
        const auto& phase = d4x4_phase_maps[phases[0U]];
        const __m256i lanes = _mm256_load_si256(reinterpret_cast<const __m256i*>(phase.lanes.data()));
        const __m256 signs_a = _mm256_load_ps(phase.signs_a.data());
        const __m256 signs_b = _mm256_load_ps(phase.signs_b.data());
        const auto emit = [&](const std::size_t output, const __m256 value, const __m256 signs) noexcept {
            const __m256 packed = _mm256_xor_ps(_mm256_permutevar8x32_ps(value, lanes), signs);
            float* const destination = block.outputs[output].data() + 2U * output_index;
            if constexpr (AlignedD4x4) {
                _mm256_store_ps(destination, packed);
            } else {
                _mm256_storeu_ps(destination, packed);
            }
        };

        emit(0U, _mm256_sub_ps(bins[2U], bins[3U]), signs_a);
        emit(1U, _mm256_sub_ps(bins[6U], bins[7U]), signs_b);
        emit(2U, _mm256_add_ps(bins[0U], bins[1U]), signs_a);
        emit(3U, _mm256_add_ps(bins[4U], bins[5U]), signs_b);
    } else {
        butterfly(bins[0U], bins[1U]);
        butterfly(bins[2U], bins[3U]);
        butterfly(bins[4U], bins[5U]);
        butterfly(bins[6U], bins[7U]);

        constexpr std::array<std::size_t, 8U> bit_reversed{0U, 4U, 2U, 6U, 1U, 5U, 3U, 7U};
        const std::size_t selected_count = data.selected_output_count();
        const std::size_t* const selected_bins = PfbChannelizerAccess::selected_fft_bins(data);
        const std::size_t phase_period = PfbChannelizerAccess::phase_period(data);
        const float* const phase_table_re = PfbChannelizerAccess::post_phase_re(data);
        const float* const phase_table_im = PfbChannelizerAccess::post_phase_im(data);
        for (std::size_t output = 0U; output < selected_count; ++output) {
            const __m256 result = bins[bit_reversed[selected_bins[output]]];
            const __m128 result_re = _mm256_castps256_ps128(result);
            const __m128 result_im = _mm256_extractf128_ps(result, 1);
            const std::size_t phase_base = output * phase_period;
            const __m128 phase_re = _mm_setr_ps(phase_table_re[phase_base + phases[0U]], phase_table_re[phase_base + phases[1U]],
                                                phase_table_re[phase_base + phases[2U]], phase_table_re[phase_base + phases[3U]]);
            const __m128 phase_im = _mm_setr_ps(phase_table_im[phase_base + phases[0U]], phase_table_im[phase_base + phases[1U]],
                                                phase_table_im[phase_base + phases[2U]], phase_table_im[phase_base + phases[3U]]);
            __m128 output_re;
            __m128 output_im;
            if (data.decimation() % 4U == 0U) {
                const __m128 sign_bit = _mm_set1_ps(-0.0f);
                const __m128 real_sign = _mm_and_ps(phase_re, sign_bit);
                const __m128 imag_sign = _mm_and_ps(phase_im, sign_bit);
                const __m128 imag_axis = _mm_cmpneq_ps(phase_im, _mm_setzero_ps());
                const __m128 real_axis_re = _mm_xor_ps(result_re, real_sign);
                const __m128 real_axis_im = _mm_xor_ps(result_im, real_sign);
                const __m128 imag_axis_re = _mm_xor_ps(result_im, _mm_xor_ps(imag_sign, sign_bit));
                const __m128 imag_axis_im = _mm_xor_ps(result_re, imag_sign);
                output_re = _mm_blendv_ps(real_axis_re, imag_axis_re, imag_axis);
                output_im = _mm_blendv_ps(real_axis_im, imag_axis_im, imag_axis);
            } else {
                const __m128 general_re = _mm_fmsub_ps(result_re, phase_re, _mm_mul_ps(result_im, phase_im));
                const __m128 general_im = _mm_fmadd_ps(result_re, phase_im, _mm_mul_ps(result_im, phase_re));
                const __m128 sign_bit = _mm_set1_ps(-0.0f);
                const __m128 one = _mm_set1_ps(1.0f);
                const __m128 zero = _mm_setzero_ps();
                const __m128 real_axis = _mm_and_ps(_mm_cmpeq_ps(_mm_andnot_ps(sign_bit, phase_re), one), _mm_cmpeq_ps(phase_im, zero));
                const __m128 imag_axis = _mm_and_ps(_mm_cmpeq_ps(phase_re, zero), _mm_cmpeq_ps(_mm_andnot_ps(sign_bit, phase_im), one));
                const __m128 real_sign = _mm_and_ps(phase_re, sign_bit);
                const __m128 imag_sign = _mm_and_ps(phase_im, sign_bit);
                const __m128 axis_re = _mm_blendv_ps(_mm_xor_ps(result_re, real_sign), _mm_xor_ps(result_im, _mm_xor_ps(imag_sign, sign_bit)), imag_axis);
                const __m128 axis_im = _mm_blendv_ps(_mm_xor_ps(result_im, real_sign), _mm_xor_ps(result_re, imag_sign), imag_axis);
                const __m128 axis = _mm_or_ps(real_axis, imag_axis);
                output_re = _mm_blendv_ps(general_re, axis_re, axis);
                output_im = _mm_blendv_ps(general_im, axis_im, axis);
            }
            const __m128 low = _mm_unpacklo_ps(output_re, output_im);
            const __m128 high = _mm_unpackhi_ps(output_re, output_im);
            float* const destination = block.outputs[output].data() + 2U * output_index;
            const __m256 packed = _mm256_insertf128_ps(_mm256_castps128_ps256(low), high, 1);
            if ((reinterpret_cast<std::uintptr_t>(destination) & 31U) == 0U) {
                _mm256_store_ps(destination, packed);
            } else {
                _mm256_storeu_ps(destination, packed);
            }
        }
    }
}

template <bool ExactD4x4, typename ProcessBatch>
[[nodiscard]] std::size_t process_streaming_d4x4(PfbChannelizerData& data, const PfbChannelizerBlock& block, ProcessBatch&& process_batch) noexcept {
    const std::size_t history_size = PfbChannelizerAccess::history_size(data);
    const std::size_t history_mask = history_size - 1U;
    const std::size_t phase_period = PfbChannelizerAccess::phase_period(data);
    float* const history_i = PfbChannelizerAccess::history_i(data);
    float* const history_q = PfbChannelizerAccess::history_q(data);
    std::size_t cursor = PfbChannelizerAccess::cursor(data);
    std::size_t decimation_phase = PfbChannelizerAccess::decimation_phase(data);
    std::size_t post_phase = PfbChannelizerAccess::post_phase(data);
    std::size_t input_index = 0U;
    std::size_t produced = 0U;
    const std::size_t input_count = block.input.size() / 2U;

    const auto write_one = [&](const std::size_t target) noexcept {
        const float sample_re = block.input[2U * input_index];
        const float sample_im = block.input[2U * input_index + 1U];
        history_i[target] = sample_re;
        history_i[target + history_size] = sample_re;
        history_q[target] = sample_im;
        history_q[target + history_size] = sample_im;
        ++input_index;
    };

    while (input_index < input_count && decimation_phase != 0U) {
        write_one(cursor);
        cursor = (cursor + 1U) & history_mask;
        decimation_phase = decimation_phase + 1U == 4U ? 0U : decimation_phase + 1U;
    }

    const auto write_contiguous = [&]<std::size_t Count>() noexcept {
        if (cursor + Count <= history_size) {
            if constexpr (Count == 16U) {
                const float* const source = block.input.data() + 2U * input_index;
                const auto deinterleave_eight = [&](const std::size_t source_offset, const std::size_t target_offset) noexcept {
                    const __m256 first = _mm256_loadu_ps(source + source_offset);
                    const __m256 second = _mm256_loadu_ps(source + source_offset + 8U);
                    const __m256 real = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(_mm256_shuffle_ps(first, second, 0x88)), 0xD8));
                    const __m256 imag = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(_mm256_shuffle_ps(first, second, 0xDD)), 0xD8));
                    _mm256_storeu_ps(history_i + target_offset, real);
                    _mm256_storeu_ps(history_i + target_offset + history_size, real);
                    _mm256_storeu_ps(history_q + target_offset, imag);
                    _mm256_storeu_ps(history_q + target_offset + history_size, imag);
                };
                deinterleave_eight(0U, cursor);
                deinterleave_eight(16U, cursor + 8U);
                input_index += Count;
            } else {
                for (std::size_t offset = 0U; offset < Count; ++offset) {
                    write_one(cursor + offset);
                }
            }
        } else {
            for (std::size_t offset = 0U; offset < Count; ++offset) {
                write_one((cursor + offset) & history_mask);
            }
        }
        cursor = (cursor + Count) & history_mask;
    };

    // Stage a run of input into the history ring first, then filter it. Interleaving the two
    // makes every row-0 load overlap stores that have not left the store buffer yet.
    constexpr std::size_t group_samples = 16U;
    constexpr std::size_t groups = pfb_write_lookahead / group_samples;
    static_assert(groups >= 1U && groups * group_samples == pfb_write_lookahead);

    std::array<std::size_t, groups> group_cursors{};
    std::array<std::array<std::size_t, 4U>, groups> group_phases{};
    while (input_count - input_index >= pfb_write_lookahead) {
        if constexpr (ExactD4x4) {
            const std::size_t lookahead_cursor = cursor;
            for (std::size_t group = 0U; group < groups; ++group) {
                write_contiguous.template operator()<group_samples>();
            }
            const std::array<std::size_t, 4U> phases{
                post_phase,
                (post_phase + 1U) & 3U,
                (post_phase + 2U) & 3U,
                (post_phase + 3U) & 3U,
            };
            for (std::size_t group = 0U; group < groups; ++group) {
                const std::size_t batch_cursor = (lookahead_cursor + group * group_samples) & history_mask;
                const std::array<std::size_t, 4U> cursors{
                    batch_cursor,
                    (batch_cursor + 4U) & history_mask,
                    (batch_cursor + 8U) & history_mask,
                    (batch_cursor + 12U) & history_mask,
                };
                process_batch(cursors.data(), phases.data(), 4U, produced);
                produced += 4U;
            }
        } else {
            for (std::size_t group = 0U; group < groups; ++group) {
                group_cursors[group] = cursor;
                write_contiguous.template operator()<group_samples>();
                for (std::size_t hop = 0U; hop < 4U; ++hop) {
                    group_phases[group][hop] = post_phase;
                    post_phase = post_phase + 1U == phase_period ? 0U : post_phase + 1U;
                }
            }
            for (std::size_t group = 0U; group < groups; ++group) {
                const std::size_t batch_cursor = group_cursors[group];
                const std::array<std::size_t, 4U> cursors{
                    batch_cursor,
                    (batch_cursor + 4U) & history_mask,
                    (batch_cursor + 8U) & history_mask,
                    (batch_cursor + 12U) & history_mask,
                };
                process_batch(cursors.data(), group_phases[group].data(), 4U, produced);
                produced += 4U;
            }
        }
    }

    // Remainder that is still a whole 16-sample group: keep the original lockstep shape so the
    // per-hop batching (and therefore the accumulation order) does not depend on where a caller
    // happens to split the stream.
    while (input_count - input_index >= group_samples) {
        const std::size_t batch_cursor = cursor;
        write_contiguous.template operator()<13U>();
        const std::array<std::size_t, 4U> cursors{
            batch_cursor,
            (batch_cursor + 4U) & history_mask,
            (batch_cursor + 8U) & history_mask,
            (batch_cursor + 12U) & history_mask,
        };
        std::array<std::size_t, 4U> phases{};
        if constexpr (ExactD4x4) {
            phases = {post_phase, (post_phase + 1U) & 3U, (post_phase + 2U) & 3U, (post_phase + 3U) & 3U};
        } else {
            for (std::size_t hop = 0U; hop < phases.size(); ++hop) {
                phases[hop] = post_phase;
                post_phase = post_phase + 1U == phase_period ? 0U : post_phase + 1U;
            }
        }
        process_batch(cursors.data(), phases.data(), 4U, produced);
        produced += 4U;
        write_contiguous.template operator()<3U>();
    }

    std::array<std::size_t, 4U> queued_cursors{};
    std::array<std::size_t, 4U> queued_phases{};
    std::size_t queued = 0U;
    while (input_index < input_count) {
        write_one(cursor);
        if (decimation_phase == 0U) {
            queued_cursors[queued] = cursor;
            queued_phases[queued] = post_phase;
            ++queued;
            post_phase = post_phase + 1U == phase_period ? 0U : post_phase + 1U;
        }
        cursor = (cursor + 1U) & history_mask;
        decimation_phase = decimation_phase + 1U == 4U ? 0U : decimation_phase + 1U;
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

template <std::size_t Bins, std::size_t HopCount, bool Direct, bool Rotate, bool RowIlp>
void process_batch_128(const PfbChannelizerData& data, const PfbChannelizerBlock& block, const std::size_t* const cursors, const std::size_t* const phases,
                       const std::size_t output_index) noexcept {
    static_assert(Bins == 4U);
    const std::size_t rows = PfbChannelizerAccess::rows(data);
    const std::size_t history_size = PfbChannelizerAccess::history_size(data);
    const float* coefficients = PfbChannelizerAccess::reversed_coefficients(data);
    const float* history_i = PfbChannelizerAccess::history_i(data);
    const float* history_q = PfbChannelizerAccess::history_q(data);
    __m128 weights_re = _mm_setzero_ps();
    __m128 weights_im = _mm_setzero_ps();
    if constexpr (Direct) {
        weights_re = _mm_load_ps(PfbChannelizerAccess::selected_transform_re(data));
        weights_im = _mm_load_ps(PfbChannelizerAccess::selected_transform_im(data));
    }
    alignas(32) std::array<float, 4U * Bins> values_re;
    alignas(32) std::array<float, 4U * Bins> values_im;
    constexpr std::size_t chain_count = RowIlp ? 4U : 1U;
    __m128 accumulator_re[4U][4U];
    __m128 accumulator_im[4U][4U];
    for (std::size_t hop = 0U; hop < HopCount; ++hop) {
        for (std::size_t chain = 0U; chain < chain_count; ++chain) {
            accumulator_re[hop][chain] = _mm_setzero_ps();
            accumulator_im[hop][chain] = _mm_setzero_ps();
        }
    }
    for (std::size_t row = 0U; row < rows; ++row) {
        const __m128 coefficient = _mm_loadu_ps(coefficients + row * Bins);
        const std::size_t row_offset = history_size - row * Bins - (Bins - 1U);
        const std::size_t chain = RowIlp ? row % chain_count : 0U;
        for (std::size_t hop = 0U; hop < HopCount; ++hop) {
            const std::size_t first_sample = cursors[hop] + row_offset;
            accumulator_re[hop][chain] = _mm_fmadd_ps(_mm_loadu_ps(history_i + first_sample), coefficient, accumulator_re[hop][chain]);
            accumulator_im[hop][chain] = _mm_fmadd_ps(_mm_loadu_ps(history_q + first_sample), coefficient, accumulator_im[hop][chain]);
        }
    }
    for (std::size_t hop = 0U; hop < HopCount; ++hop) {
        __m128 accumulated_re = accumulator_re[hop][0U];
        __m128 accumulated_im = accumulator_im[hop][0U];
        if constexpr (RowIlp) {
            accumulated_re = _mm_add_ps(_mm_add_ps(accumulated_re, accumulator_re[hop][1U]), _mm_add_ps(accumulator_re[hop][2U], accumulator_re[hop][3U]));
            accumulated_im = _mm_add_ps(_mm_add_ps(accumulated_im, accumulator_im[hop][1U]), _mm_add_ps(accumulator_im[hop][2U], accumulator_im[hop][3U]));
        }
        const __m128 natural_re = reverse_lanes(accumulated_re);
        const __m128 natural_im = reverse_lanes(accumulated_im);
        __m128 transformed_re = natural_re;
        __m128 transformed_im = natural_im;
        if constexpr (Rotate) {
            const __m128 rotation_re = _mm_load_ps(PfbChannelizerAccess::branch_rotation_re(data));
            const __m128 rotation_im = _mm_load_ps(PfbChannelizerAccess::branch_rotation_im(data));
            transformed_re = _mm_fmsub_ps(natural_re, rotation_re, _mm_mul_ps(natural_im, rotation_im));
            transformed_im = _mm_fmadd_ps(natural_re, rotation_im, _mm_mul_ps(natural_im, rotation_re));
        }
        if constexpr (Direct) {
            const __m128 value_re = _mm_fmsub_ps(transformed_re, weights_re, _mm_mul_ps(transformed_im, weights_im));
            const __m128 value_im = _mm_fmadd_ps(transformed_re, weights_im, _mm_mul_ps(transformed_im, weights_re));
            pfb_store_output(block.outputs[0], output_index + hop,
                             pfb_apply_post_phase(data, 0U, phases[hop], horizontal_sum(value_re), horizontal_sum(value_im)));
            continue;
        }
        _mm_store_ps(values_re.data() + hop * Bins, transformed_re);
        _mm_store_ps(values_im.data() + hop * Bins, transformed_im);
    }
    if constexpr (!Direct) {
        Ifft_avx2_fma(values_re.data(), values_im.data(), Bins, HopCount, Bins);
        for (std::size_t hop = 0U; hop < HopCount; ++hop) {
            pfb_emit_transformed_outputs(data, block, output_index + hop, phases[hop], values_re.data() + hop * Bins, values_im.data() + hop * Bins);
        }
    }
}

template <std::size_t Bins, std::size_t HopCount, bool Direct, bool Rotate, bool RowIlp, bool ExactD4x4, bool AlignedD4x4, std::size_t FixedRows>
void process_batch_256(const PfbChannelizerData& data, const PfbChannelizerBlock& block, const std::size_t* const cursors, const std::size_t* const phases,
                       const std::size_t output_index) noexcept {
    static_assert(Bins >= 8U && Bins % 8U == 0U);
    static_assert(!ExactD4x4 || (Bins == 8U && !Direct && Rotate && !RowIlp));
    static_assert(FixedRows == 0U || (ExactD4x4 && Bins == 8U));
    constexpr std::size_t width = 8U;
    constexpr std::size_t chunk_count = Bins / width;
    const std::size_t rows = FixedRows == 0U ? PfbChannelizerAccess::rows(data) : FixedRows;
    const std::size_t history_size = PfbChannelizerAccess::history_size(data);
    const float* coefficients = PfbChannelizerAccess::reversed_coefficients(data);
    const float* history_i = PfbChannelizerAccess::history_i(data);
    const float* history_q = PfbChannelizerAccess::history_q(data);
    const float* rotations_re = PfbChannelizerAccess::branch_rotation_re(data);
    const float* rotations_im = PfbChannelizerAccess::branch_rotation_im(data);
    const float* weights_re = PfbChannelizerAccess::selected_transform_re(data);
    const float* weights_im = PfbChannelizerAccess::selected_transform_im(data);
    alignas(32) std::array<float, 4U * Bins> values_re;
    alignas(32) std::array<float, 4U * Bins> values_im;
    __m256 batch_re[4U];
    __m256 batch_im[4U];
    __m256 direct_re[4U];
    __m256 direct_im[4U];
    if constexpr (Direct) {
        for (std::size_t hop = 0U; hop < HopCount; ++hop) {
            direct_re[hop] = _mm256_setzero_ps();
            direct_im[hop] = _mm256_setzero_ps();
        }
    }

    for (std::size_t destination_chunk = 0U; destination_chunk < chunk_count; ++destination_chunk) {
        const std::size_t source_chunk = chunk_count - 1U - destination_chunk;
        constexpr std::size_t chain_count = RowIlp ? 4U : 1U;
        __m256 accumulator_re[4U][4U];
        __m256 accumulator_im[4U][4U];
        for (std::size_t hop = 0U; hop < HopCount; ++hop) {
            for (std::size_t chain = 0U; chain < chain_count; ++chain) {
                accumulator_re[hop][chain] = _mm256_setzero_ps();
                accumulator_im[hop][chain] = _mm256_setzero_ps();
            }
        }
        const std::size_t chunk_offset = source_chunk * width;
        if constexpr (Bins == 8U && HopCount == 4U && !RowIlp) {
            if constexpr (ExactD4x4) {
                std::size_t row = 0U;
                for (; row + 1U < rows; row += 2U) {
                    const __m256 coefficient0 = _mm256_load_ps(coefficients + row * Bins);
                    const __m256 coefficient1 = _mm256_load_ps(coefficients + (row + 1U) * Bins);
                    const std::size_t offset0 = history_size - row * Bins - (Bins - 1U);
                    const std::size_t offset1 = offset0 - Bins;
                    const auto accumulate_pair = [&](const float* const history, auto& accumulators) noexcept {
                        const __m256 a0 = _mm256_loadu_ps(history + cursors[0U] + offset0);
                        const __m256 a1 = _mm256_loadu_ps(history + cursors[1U] + offset0);
                        const __m256 a2 = _mm256_loadu_ps(history + cursors[2U] + offset0);
                        const __m256 a3 = _mm256_loadu_ps(history + cursors[3U] + offset0);
                        const __m256 b0 = _mm256_loadu_ps(history + cursors[0U] + offset1);
                        const __m256 b1 = _mm256_loadu_ps(history + cursors[1U] + offset1);
                        accumulators[0U][0U] = _mm256_fmadd_ps(a0, coefficient0, accumulators[0U][0U]);
                        accumulators[0U][0U] = _mm256_fmadd_ps(b0, coefficient1, accumulators[0U][0U]);
                        accumulators[1U][0U] = _mm256_fmadd_ps(a1, coefficient0, accumulators[1U][0U]);
                        accumulators[1U][0U] = _mm256_fmadd_ps(b1, coefficient1, accumulators[1U][0U]);
                        accumulators[2U][0U] = _mm256_fmadd_ps(a2, coefficient0, accumulators[2U][0U]);
                        accumulators[2U][0U] = _mm256_fmadd_ps(a0, coefficient1, accumulators[2U][0U]);
                        accumulators[3U][0U] = _mm256_fmadd_ps(a3, coefficient0, accumulators[3U][0U]);
                        accumulators[3U][0U] = _mm256_fmadd_ps(a1, coefficient1, accumulators[3U][0U]);
                    };
                    accumulate_pair(history_i, accumulator_re);
                    accumulate_pair(history_q, accumulator_im);
                }
                if (row != rows) {
                    const __m256 coefficient = _mm256_load_ps(coefficients + row * Bins);
                    const std::size_t row_offset = history_size - row * Bins - (Bins - 1U);
                    for (std::size_t hop = 0U; hop < HopCount; ++hop) {
                        const std::size_t first_sample = cursors[hop] + row_offset;
                        accumulator_re[hop][0U] = _mm256_fmadd_ps(_mm256_loadu_ps(history_i + first_sample), coefficient, accumulator_re[hop][0U]);
                        accumulator_im[hop][0U] = _mm256_fmadd_ps(_mm256_loadu_ps(history_q + first_sample), coefficient, accumulator_im[hop][0U]);
                    }
                }
            } else if (data.decimation() == 4U) {
                std::size_t row = 0U;
                for (; row + 1U < rows; row += 2U) {
                    const __m256 coefficient0 = _mm256_load_ps(coefficients + row * Bins);
                    const __m256 coefficient1 = _mm256_load_ps(coefficients + (row + 1U) * Bins);
                    const std::size_t offset0 = history_size - row * Bins - (Bins - 1U);
                    const std::size_t offset1 = offset0 - Bins;
                    const auto accumulate_pair = [&](const float* const history, auto& accumulators) noexcept {
                        const __m256 a0 = _mm256_loadu_ps(history + cursors[0U] + offset0);
                        const __m256 a1 = _mm256_loadu_ps(history + cursors[1U] + offset0);
                        const __m256 a2 = _mm256_loadu_ps(history + cursors[2U] + offset0);
                        const __m256 a3 = _mm256_loadu_ps(history + cursors[3U] + offset0);
                        const __m256 b0 = _mm256_loadu_ps(history + cursors[0U] + offset1);
                        const __m256 b1 = _mm256_loadu_ps(history + cursors[1U] + offset1);
                        accumulators[0U][0U] = _mm256_fmadd_ps(a0, coefficient0, accumulators[0U][0U]);
                        accumulators[0U][0U] = _mm256_fmadd_ps(b0, coefficient1, accumulators[0U][0U]);
                        accumulators[1U][0U] = _mm256_fmadd_ps(a1, coefficient0, accumulators[1U][0U]);
                        accumulators[1U][0U] = _mm256_fmadd_ps(b1, coefficient1, accumulators[1U][0U]);
                        accumulators[2U][0U] = _mm256_fmadd_ps(a2, coefficient0, accumulators[2U][0U]);
                        accumulators[2U][0U] = _mm256_fmadd_ps(a0, coefficient1, accumulators[2U][0U]);
                        accumulators[3U][0U] = _mm256_fmadd_ps(a3, coefficient0, accumulators[3U][0U]);
                        accumulators[3U][0U] = _mm256_fmadd_ps(a1, coefficient1, accumulators[3U][0U]);
                    };
                    accumulate_pair(history_i, accumulator_re);
                    accumulate_pair(history_q, accumulator_im);
                }
                if (row != rows) {
                    const __m256 coefficient = _mm256_load_ps(coefficients + row * Bins);
                    const std::size_t row_offset = history_size - row * Bins - (Bins - 1U);
                    for (std::size_t hop = 0U; hop < HopCount; ++hop) {
                        const std::size_t first_sample = cursors[hop] + row_offset;
                        accumulator_re[hop][0U] = _mm256_fmadd_ps(_mm256_loadu_ps(history_i + first_sample), coefficient, accumulator_re[hop][0U]);
                        accumulator_im[hop][0U] = _mm256_fmadd_ps(_mm256_loadu_ps(history_q + first_sample), coefficient, accumulator_im[hop][0U]);
                    }
                }
            } else {
                for (std::size_t row = 0U; row < rows; ++row) {
                    const __m256 coefficient = _mm256_load_ps(coefficients + row * Bins);
                    const std::size_t row_offset = history_size - row * Bins - (Bins - 1U);
                    for (std::size_t hop = 0U; hop < HopCount; ++hop) {
                        const std::size_t first_sample = cursors[hop] + row_offset;
                        accumulator_re[hop][0U] = _mm256_fmadd_ps(_mm256_loadu_ps(history_i + first_sample), coefficient, accumulator_re[hop][0U]);
                        accumulator_im[hop][0U] = _mm256_fmadd_ps(_mm256_loadu_ps(history_q + first_sample), coefficient, accumulator_im[hop][0U]);
                    }
                }
            }
        } else {
            for (std::size_t row = 0U; row < rows; ++row) {
                const __m256 coefficient = _mm256_load_ps(coefficients + row * Bins + chunk_offset);
                const std::size_t row_offset = history_size - row * Bins - (Bins - 1U) + chunk_offset;
                const std::size_t chain = RowIlp ? row % chain_count : 0U;
                for (std::size_t hop = 0U; hop < HopCount; ++hop) {
                    const std::size_t first_sample = cursors[hop] + row_offset;
                    accumulator_re[hop][chain] = _mm256_fmadd_ps(_mm256_loadu_ps(history_i + first_sample), coefficient, accumulator_re[hop][chain]);
                    accumulator_im[hop][chain] = _mm256_fmadd_ps(_mm256_loadu_ps(history_q + first_sample), coefficient, accumulator_im[hop][chain]);
                }
            }
        }

        const std::size_t destination_offset = destination_chunk * width;
        __m256 weight_re = _mm256_setzero_ps();
        __m256 weight_im = _mm256_setzero_ps();
        if constexpr (Direct) {
            weight_re = _mm256_load_ps(weights_re + destination_offset);
            weight_im = _mm256_load_ps(weights_im + destination_offset);
        }
        for (std::size_t hop = 0U; hop < HopCount; ++hop) {
            __m256 accumulated_re = accumulator_re[hop][0U];
            __m256 accumulated_im = accumulator_im[hop][0U];
            if constexpr (RowIlp) {
                accumulated_re =
                    _mm256_add_ps(_mm256_add_ps(accumulated_re, accumulator_re[hop][1U]), _mm256_add_ps(accumulator_re[hop][2U], accumulator_re[hop][3U]));
                accumulated_im =
                    _mm256_add_ps(_mm256_add_ps(accumulated_im, accumulator_im[hop][1U]), _mm256_add_ps(accumulator_im[hop][2U], accumulator_im[hop][3U]));
            }
            const __m256 natural_re = reverse_lanes(accumulated_re);
            const __m256 natural_im = reverse_lanes(accumulated_im);
            __m256 transformed_re = natural_re;
            __m256 transformed_im = natural_im;
            if constexpr (Rotate) {
                const __m256 rotation_re = _mm256_load_ps(rotations_re + destination_offset);
                const __m256 rotation_im = _mm256_load_ps(rotations_im + destination_offset);
                transformed_re = _mm256_fmsub_ps(natural_re, rotation_re, _mm256_mul_ps(natural_im, rotation_im));
                transformed_im = _mm256_fmadd_ps(natural_re, rotation_im, _mm256_mul_ps(natural_im, rotation_re));
            }
            if constexpr (Direct) {
                direct_re[hop] = _mm256_fmadd_ps(transformed_re, weight_re, direct_re[hop]);
                direct_re[hop] = _mm256_fnmadd_ps(transformed_im, weight_im, direct_re[hop]);
                direct_im[hop] = _mm256_fmadd_ps(transformed_re, weight_im, direct_im[hop]);
                direct_im[hop] = _mm256_fmadd_ps(transformed_im, weight_re, direct_im[hop]);
            } else {
                if constexpr (Bins == 8U && HopCount == 4U) {
                    batch_re[hop] = transformed_re;
                    batch_im[hop] = transformed_im;
                } else {
                    _mm256_store_ps(values_re.data() + hop * Bins + destination_offset, transformed_re);
                    _mm256_store_ps(values_im.data() + hop * Bins + destination_offset, transformed_im);
                }
            }
        }
    }

    if constexpr (Direct) {
        for (std::size_t hop = 0U; hop < HopCount; ++hop) {
            pfb_store_output(block.outputs[0], output_index + hop,
                             pfb_apply_post_phase(data, 0U, phases[hop], horizontal_sum(direct_re[hop]), horizontal_sum(direct_im[hop])));
        }
    } else {
        if constexpr (Bins == 8U && HopCount == 4U) {
            ifft8_four_hops_emit<ExactD4x4, AlignedD4x4>(data, block, output_index, phases, batch_re, batch_im);
        } else {
            Ifft_avx2_fma(values_re.data(), values_im.data(), Bins, HopCount, Bins);
            for (std::size_t hop = 0U; hop < HopCount; ++hop) {
                pfb_emit_transformed_outputs(data, block, output_index + hop, phases[hop], values_re.data() + hop * Bins, values_im.data() + hop * Bins);
            }
        }
    }
}

template <std::size_t Bins, std::size_t HopCount, bool Direct, bool Rotate, bool RowIlp = false, bool ExactD4x4 = false, bool AlignedD4x4 = false,
          std::size_t FixedRows = 0U>
void process_batch(const PfbChannelizerData& data, const PfbChannelizerBlock& block, const std::size_t* const cursors, const std::size_t* const phases,
                   const std::size_t output_index) noexcept {
    if constexpr (Bins == 4U) {
        process_batch_128<Bins, HopCount, Direct, Rotate, RowIlp>(data, block, cursors, phases, output_index);
    } else {
        process_batch_256<Bins, HopCount, Direct, Rotate, RowIlp, ExactD4x4, AlignedD4x4, FixedRows>(data, block, cursors, phases, output_index);
    }
}

template <std::size_t Bins, bool Direct, bool Rotate, bool ExactD4x4 = false, bool AlignedD4x4 = false, std::size_t FixedRows = 0U>
[[nodiscard]] std::size_t process(PfbChannelizerData& data, const PfbChannelizerBlock& block) noexcept {
    static_assert(!ExactD4x4 || (Bins == 8U && !Direct && Rotate));
    static_assert(FixedRows == 0U || ExactD4x4);
    const std::size_t filter_span = PfbChannelizerAccess::rows(data) * Bins;
    const std::size_t history_size = PfbChannelizerAccess::history_size(data);
    const std::size_t batch_limit = std::min<std::size_t>(4U, 1U + (history_size - filter_span) / data.decimation());
    const auto dispatch_batch = [&](const std::size_t* const cursors, const std::size_t* const phases, const std::size_t hop_count,
                                    const std::size_t output_index) noexcept {
        if (hop_count == 1U) {
            process_batch<Bins, 1U, Direct, Rotate, true, false>(data, block, cursors, phases, output_index);
            return;
        }
        switch (hop_count) {
        case 1U:
            process_batch<Bins, 1U, Direct, Rotate, false, ExactD4x4, AlignedD4x4, FixedRows>(data, block, cursors, phases, output_index);
            break;
        case 2U:
            process_batch<Bins, 2U, Direct, Rotate, false, ExactD4x4, AlignedD4x4, FixedRows>(data, block, cursors, phases, output_index);
            break;
        case 3U:
            process_batch<Bins, 3U, Direct, Rotate, false, ExactD4x4, AlignedD4x4, FixedRows>(data, block, cursors, phases, output_index);
            break;
        case 4U:
            process_batch<Bins, 4U, Direct, Rotate, false, ExactD4x4, AlignedD4x4, FixedRows>(data, block, cursors, phases, output_index);
            break;
        default:
            break;
        }
    };
    if constexpr (ExactD4x4) {
        return process_streaming_d4x4<true>(data, block, dispatch_batch);
    } else if constexpr (Bins == 8U && !Direct) {
        if (data.decimation() == 4U && data.selected_output_count() != 0U) {
            return process_streaming_d4x4<false>(data, block, dispatch_batch);
        }
    }
    return pfb_process_streaming(data, block, batch_limit, dispatch_batch);
}

template <std::size_t Bins> [[nodiscard]] std::size_t process_selected(PfbChannelizerData& data, const PfbChannelizerBlock& block) noexcept {
    if (data.selected_output_count() == 1U) {
        return process<Bins, true, false>(data, block);
    }
    if (data.grid_offset() == PfbGridOffset::half_bins) {
        return process<Bins, false, true>(data, block);
    }
    return process<Bins, false, false>(data, block);
}

} // namespace

std::size_t PfbChannelizer_avx2fma(PfbChannelizerData& data, const PfbChannelizerBlock& block) noexcept {
    std::size_t produced = 0U;
    switch (data.bin_count()) {
    case 4U:
        produced = process_selected<4U>(data, block);
        break;
    case 8U: {
        const auto logical_bins = data.logical_bins();
        const bool exact_d4x4 = data.decimation() == 4U && data.grid_offset() == PfbGridOffset::half_bins && logical_bins.size() == 4U &&
                                logical_bins[0U] == -2 && logical_bins[1U] == -1 && logical_bins[2U] == 0 && logical_bins[3U] == 1;
        bool aligned_d4x4 = exact_d4x4;
        for (std::size_t output = 0U; output < 4U && aligned_d4x4; ++output) {
            aligned_d4x4 = (reinterpret_cast<std::uintptr_t>(block.outputs[output].data()) & 31U) == 0U;
        }
        const bool fixed_rows = aligned_d4x4 && PfbChannelizerAccess::rows(data) == 22U;
        produced = exact_d4x4 ? (aligned_d4x4
                                     ? (fixed_rows ? process<8U, false, true, true, true, 22U>(data, block) : process<8U, false, true, true, true>(data, block))
                                     : process<8U, false, true, true, false>(data, block))
                              : process_selected<8U>(data, block);
    } break;
    case 16U:
        produced = process_selected<16U>(data, block);
        break;
    case 32U:
        produced = process_selected<32U>(data, block);
        break;
    default:
        return PfbChannelizer_generic(data, block);
    }
    return produced;
}

} // namespace uni::simd::detail
