#include "pfb_channelizer/pfb_channelizer_internal.hpp"
#include "ifft_cf32/ifft_cf32_internal.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

#include <immintrin.h>

namespace uni::simd::detail {
#if UNI_SIMD_HAVE_AVX2_FMA
namespace {

[[nodiscard]] inline __m512 reverse_lanes(const __m512 value) noexcept {
    const __m512i reverse = _mm512_setr_epi32(
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    return _mm512_permutexvar_ps(reverse, value);
}

template <std::size_t Bins, std::size_t HopCount, bool Direct, bool Rotate, bool RowIlp>
void process_batch(const PfbChannelizerData& data, const PfbChannelizerBlock& block,
                   const std::size_t* const cursors, const std::size_t* const phases,
                   const std::size_t output_index) noexcept {
    static_assert(Bins == 16U || Bins == 32U);
    constexpr std::size_t width = 16U;
    constexpr std::size_t chunk_count = Bins / width;
    constexpr std::size_t chain_count = RowIlp ? 4U : 1U;
    const std::size_t rows = PfbChannelizerAccess::rows(data);
    const std::size_t history_size = PfbChannelizerAccess::history_size(data);
    const float* coefficients = PfbChannelizerAccess::reversed_coefficients(data);
    const float* history_i = PfbChannelizerAccess::history_i(data);
    const float* history_q = PfbChannelizerAccess::history_q(data);
    const float* rotations_re = PfbChannelizerAccess::branch_rotation_re(data);
    const float* rotations_im = PfbChannelizerAccess::branch_rotation_im(data);
    const float* weights_re = PfbChannelizerAccess::selected_transform_re(data);
    const float* weights_im = PfbChannelizerAccess::selected_transform_im(data);
    alignas(64) std::array<float, 4U * Bins> values_re;
    alignas(64) std::array<float, 4U * Bins> values_im;
    __m512 direct_re[4U];
    __m512 direct_im[4U];
    if constexpr (Direct) {
        for (std::size_t hop = 0U; hop < HopCount; ++hop) {
            direct_re[hop] = _mm512_setzero_ps();
            direct_im[hop] = _mm512_setzero_ps();
        }
    }

    for (std::size_t destination_chunk = 0U; destination_chunk < chunk_count; ++destination_chunk) {
        const std::size_t source_chunk = chunk_count - 1U - destination_chunk;
        __m512 accumulator_re[4U][4U];
        __m512 accumulator_im[4U][4U];
        for (std::size_t hop = 0U; hop < HopCount; ++hop) {
            for (std::size_t chain = 0U; chain < chain_count; ++chain) {
                accumulator_re[hop][chain] = _mm512_setzero_ps();
                accumulator_im[hop][chain] = _mm512_setzero_ps();
            }
        }
        for (std::size_t row = 0U; row < rows; ++row) {
            const std::size_t chunk_offset = source_chunk * width;
            const __m512 coefficient = _mm512_load_ps(coefficients + row * Bins + chunk_offset);
            const std::size_t row_offset = history_size - row * Bins - (Bins - 1U) + chunk_offset;
            const std::size_t chain = RowIlp ? row % chain_count : 0U;
            for (std::size_t hop = 0U; hop < HopCount; ++hop) {
                const std::size_t first_sample = cursors[hop] + row_offset;
                accumulator_re[hop][chain] = _mm512_fmadd_ps(
                    _mm512_loadu_ps(history_i + first_sample), coefficient, accumulator_re[hop][chain]);
                accumulator_im[hop][chain] = _mm512_fmadd_ps(
                    _mm512_loadu_ps(history_q + first_sample), coefficient, accumulator_im[hop][chain]);
            }
        }

        const std::size_t destination_offset = destination_chunk * width;
        __m512 weight_re = _mm512_setzero_ps();
        __m512 weight_im = _mm512_setzero_ps();
        if constexpr (Direct) {
            weight_re = _mm512_load_ps(weights_re + destination_offset);
            weight_im = _mm512_load_ps(weights_im + destination_offset);
        }
        for (std::size_t hop = 0U; hop < HopCount; ++hop) {
            __m512 accumulated_re = accumulator_re[hop][0U];
            __m512 accumulated_im = accumulator_im[hop][0U];
            if constexpr (RowIlp) {
                accumulated_re = _mm512_add_ps(
                    _mm512_add_ps(accumulated_re, accumulator_re[hop][1U]),
                    _mm512_add_ps(accumulator_re[hop][2U], accumulator_re[hop][3U]));
                accumulated_im = _mm512_add_ps(
                    _mm512_add_ps(accumulated_im, accumulator_im[hop][1U]),
                    _mm512_add_ps(accumulator_im[hop][2U], accumulator_im[hop][3U]));
            }
            const __m512 natural_re = reverse_lanes(accumulated_re);
            const __m512 natural_im = reverse_lanes(accumulated_im);
            __m512 transformed_re = natural_re;
            __m512 transformed_im = natural_im;
            if constexpr (Rotate) {
                const __m512 rotation_re = _mm512_load_ps(rotations_re + destination_offset);
                const __m512 rotation_im = _mm512_load_ps(rotations_im + destination_offset);
                transformed_re = _mm512_fmsub_ps(
                    natural_re, rotation_re, _mm512_mul_ps(natural_im, rotation_im));
                transformed_im = _mm512_fmadd_ps(
                    natural_re, rotation_im, _mm512_mul_ps(natural_im, rotation_re));
            }
            if constexpr (Direct) {
                direct_re[hop] = _mm512_fmadd_ps(transformed_re, weight_re, direct_re[hop]);
                direct_re[hop] = _mm512_fnmadd_ps(transformed_im, weight_im, direct_re[hop]);
                direct_im[hop] = _mm512_fmadd_ps(transformed_re, weight_im, direct_im[hop]);
                direct_im[hop] = _mm512_fmadd_ps(transformed_im, weight_re, direct_im[hop]);
            } else {
                _mm512_store_ps(values_re.data() + hop * Bins + destination_offset, transformed_re);
                _mm512_store_ps(values_im.data() + hop * Bins + destination_offset, transformed_im);
            }
        }
    }

    if constexpr (Direct) {
        for (std::size_t hop = 0U; hop < HopCount; ++hop) {
            pfb_store_output(block.outputs[0], output_index + hop,
                             pfb_apply_post_phase(data, 0U, phases[hop],
                                                  _mm512_reduce_add_ps(direct_re[hop]),
                                                  _mm512_reduce_add_ps(direct_im[hop])));
        }
    } else {
        Ifft_avx2_fma(values_re.data(), values_im.data(), Bins, HopCount, Bins);
        for (std::size_t hop = 0U; hop < HopCount; ++hop) {
            pfb_emit_transformed_outputs(data, block, output_index + hop, phases[hop],
                                         values_re.data() + hop * Bins,
                                         values_im.data() + hop * Bins);
        }
    }
}

template <std::size_t Bins, bool Direct, bool Rotate>
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
                process_batch<Bins, 1U, Direct, Rotate, true>(data, block, cursors, phases, output_index);
                return;
            }
            switch (hop_count) {
            case 1U:
                process_batch<Bins, 1U, Direct, Rotate, false>(data, block, cursors, phases, output_index);
                break;
            case 2U:
                process_batch<Bins, 2U, Direct, Rotate, false>(data, block, cursors, phases, output_index);
                break;
            case 3U:
                process_batch<Bins, 3U, Direct, Rotate, false>(data, block, cursors, phases, output_index);
                break;
            case 4U:
                process_batch<Bins, 4U, Direct, Rotate, false>(data, block, cursors, phases, output_index);
                break;
            default:
                break;
            }
        });
}

template <std::size_t Bins>
[[nodiscard]] std::size_t process_selected(PfbChannelizerData& data,
                                            const PfbChannelizerBlock& block) noexcept {
    if (data.selected_output_count() == 1U) {
        return process<Bins, true, false>(data, block);
    }
    if (data.grid_offset() == PfbGridOffset::half_bins) {
        return process<Bins, false, true>(data, block);
    }
    return process<Bins, false, false>(data, block);
}

} // namespace

bool PfbChannelizer_supports_avx512(const PfbChannelizerData& data) noexcept {
    return data.bin_count() == 32U;
}

std::size_t PfbChannelizer_avx512(PfbChannelizerData& data,
                                  const PfbChannelizerBlock& block) noexcept {
    switch (data.bin_count()) {
    case 16U:
        return process_selected<16U>(data, block);
    case 32U:
        return process_selected<32U>(data, block);
    default:
        return PfbChannelizer_avx2fma(data, block);
    }
}

#else

bool PfbChannelizer_supports_avx512(const PfbChannelizerData&) noexcept {
    return false;
}

std::size_t PfbChannelizer_avx512(PfbChannelizerData& data,
                                  const PfbChannelizerBlock& block) noexcept {
    return PfbChannelizer_generic(data, block);
}

#endif

} // namespace uni::simd::detail
