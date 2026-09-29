#include "pfb_channelizer/pfb_channelizer_internal.hpp"
#include "ifft_cf32/ifft_cf32_internal.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include <immintrin.h>

namespace uni::simd::detail {
#if UNI_SIMD_HAVE_AVX2_FMA
namespace {

[[nodiscard]] inline __m512 reverse_lanes(const __m512 value) noexcept {
    const __m512i reverse = _mm512_setr_epi32(
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    return _mm512_permutexvar_ps(reverse, value);
}

[[nodiscard]] inline __m512 reverse_8_lane_halves(const __m512 value) noexcept {
    const __m512i reverse = _mm512_setr_epi32(
        7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
    return _mm512_permutexvar_ps(reverse, value);
}

template <std::size_t HopCount, bool Rotate>
void process_batch_8(const PfbChannelizerData& data, const PfbChannelizerBlock& block,
                     const std::size_t* const cursors, const std::size_t* const phases,
                     const std::size_t output_index) noexcept {
    static_assert(HopCount >= 1U && HopCount <= 4U);
    constexpr std::size_t pair_count = (HopCount + 1U) / 2U;
    constexpr __mmask16 low_lanes = 0x00ffU;
    constexpr __mmask16 high_lanes = 0xff00U;
    const std::size_t rows = PfbChannelizerAccess::rows(data);
    const std::size_t history_size = PfbChannelizerAccess::history_size(data);
    const float* coefficients = PfbChannelizerAccess::avx512_coefficients_8(data);
    const float* history_i = PfbChannelizerAccess::history_i(data);
    const float* history_q = PfbChannelizerAccess::history_q(data);
    const __m512 rotations_re = _mm512_load_ps(PfbChannelizerAccess::avx512_rotation_re_8(data));
    const __m512 rotations_im = _mm512_load_ps(PfbChannelizerAccess::avx512_rotation_im_8(data));
    alignas(64) std::array<float, 4U * 8U> values_re{};
    alignas(64) std::array<float, 4U * 8U> values_im{};
    __m512 accumulator_re[pair_count];
    __m512 accumulator_im[pair_count];
    for (std::size_t pair = 0U; pair < pair_count; ++pair) {
        accumulator_re[pair] = _mm512_setzero_ps();
        accumulator_im[pair] = _mm512_setzero_ps();
    }

    for (std::size_t row = 0U; row < rows; ++row) {
        const __m512 coefficient = _mm512_load_ps(coefficients + row * 16U);
        const std::size_t row_offset = history_size - row * 8U - 7U;
        for (std::size_t pair = 0U; pair < pair_count; ++pair) {
            const std::size_t first_hop = pair * 2U;
            __m512 samples_re = _mm512_maskz_expandloadu_ps(
                low_lanes, history_i + cursors[first_hop] + row_offset);
            __m512 samples_im = _mm512_maskz_expandloadu_ps(
                low_lanes, history_q + cursors[first_hop] + row_offset);
            if (first_hop + 1U < HopCount) {
                samples_re = _mm512_mask_expandloadu_ps(
                    samples_re, high_lanes, history_i + cursors[first_hop + 1U] + row_offset);
                samples_im = _mm512_mask_expandloadu_ps(
                    samples_im, high_lanes, history_q + cursors[first_hop + 1U] + row_offset);
            }
            accumulator_re[pair] = _mm512_fmadd_ps(samples_re, coefficient, accumulator_re[pair]);
            accumulator_im[pair] = _mm512_fmadd_ps(samples_im, coefficient, accumulator_im[pair]);
        }
    }

    for (std::size_t pair = 0U; pair < pair_count; ++pair) {
        __m512 transformed_re = reverse_8_lane_halves(accumulator_re[pair]);
        __m512 transformed_im = reverse_8_lane_halves(accumulator_im[pair]);
        if constexpr (Rotate) {
            const __m512 natural_re = transformed_re;
            const __m512 natural_im = transformed_im;
            transformed_re = _mm512_fmsub_ps(
                natural_re, rotations_re, _mm512_mul_ps(natural_im, rotations_im));
            transformed_im = _mm512_fmadd_ps(
                natural_re, rotations_im, _mm512_mul_ps(natural_im, rotations_re));
        }
        const std::size_t first_hop = pair * 2U;
        if (first_hop + 1U < HopCount) {
            _mm512_store_ps(values_re.data() + first_hop * 8U, transformed_re);
            _mm512_store_ps(values_im.data() + first_hop * 8U, transformed_im);
        } else {
            _mm512_mask_store_ps(values_re.data() + first_hop * 8U, low_lanes, transformed_re);
            _mm512_mask_store_ps(values_im.data() + first_hop * 8U, low_lanes, transformed_im);
        }
    }

    Ifft_avx2_fma(values_re.data(), values_im.data(), 8U, HopCount, 8U);
    for (std::size_t hop = 0U; hop < HopCount; ++hop) {
        pfb_emit_transformed_outputs(data, block, output_index + hop, phases[hop],
                                     values_re.data() + hop * 8U,
                                     values_im.data() + hop * 8U);
    }
}

template <bool Rotate>
[[nodiscard]] std::size_t process_8(PfbChannelizerData& data,
                                    const PfbChannelizerBlock& block) noexcept {
    return pfb_process_streaming(
        data, block, 4U,
        [&](const std::size_t* const cursors, const std::size_t* const phases,
            const std::size_t hop_count, const std::size_t output_index) noexcept {
            switch (hop_count) {
            case 1U:
                process_batch_8<1U, Rotate>(data, block, cursors, phases, output_index);
                break;
            case 2U:
                process_batch_8<2U, Rotate>(data, block, cursors, phases, output_index);
                break;
            case 3U:
                process_batch_8<3U, Rotate>(data, block, cursors, phases, output_index);
                break;
            case 4U:
                process_batch_8<4U, Rotate>(data, block, cursors, phases, output_index);
                break;
            default:
                break;
            }
        });
}

template <std::size_t Bins, std::size_t HopCount, bool Direct, bool Rotate>
void process_batch(const PfbChannelizerData& data, const PfbChannelizerBlock& block, const std::size_t* const cursors, const std::size_t* const phases,
                   const std::size_t output_index) noexcept {
    static_assert(Bins == 16U || Bins == 32U);
    constexpr std::size_t width = 16U;
    constexpr std::size_t chunk_count = Bins / width;
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
        __m512 accumulator_re[4U];
        __m512 accumulator_im[4U];
        for (std::size_t hop = 0U; hop < HopCount; ++hop) {
            accumulator_re[hop] = _mm512_setzero_ps();
            accumulator_im[hop] = _mm512_setzero_ps();
        }
        for (std::size_t row = 0U; row < rows; ++row) {
            const std::size_t chunk_offset = source_chunk * width;
            const __m512 coefficient = _mm512_load_ps(coefficients + row * Bins + chunk_offset);
            const std::size_t row_offset = history_size - row * Bins - (Bins - 1U) + chunk_offset;
            for (std::size_t hop = 0U; hop < HopCount; ++hop) {
                const std::size_t first_sample = cursors[hop] + row_offset;
                accumulator_re[hop] = _mm512_fmadd_ps(_mm512_loadu_ps(history_i + first_sample), coefficient, accumulator_re[hop]);
                accumulator_im[hop] = _mm512_fmadd_ps(_mm512_loadu_ps(history_q + first_sample), coefficient, accumulator_im[hop]);
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
            const __m512 accumulated_re = accumulator_re[hop];
            const __m512 accumulated_im = accumulator_im[hop];
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
            switch (hop_count) {
            case 1U:
                process_batch<Bins, 1U, Direct, Rotate>(data, block, cursors, phases, output_index);
                break;
            case 2U:
                process_batch<Bins, 2U, Direct, Rotate>(data, block, cursors, phases, output_index);
                break;
            case 3U:
                process_batch<Bins, 3U, Direct, Rotate>(data, block, cursors, phases, output_index);
                break;
            case 4U:
                process_batch<Bins, 4U, Direct, Rotate>(data, block, cursors, phases, output_index);
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

#if defined(__GNUC__) || defined(__clang__)
#define UNI_SIMD_D4X4_INLINE [[gnu::always_inline]] inline
#elif defined(_MSC_VER)
#define UNI_SIMD_D4X4_INLINE __forceinline
#else
#define UNI_SIMD_D4X4_INLINE inline
#endif

// Exact half-bin, 8-bin, decimate-by-4 main loop.
//
// Mirrors the AVX2 d4x4 kernel lane for lane in 512-bit registers. Hops h and h + 2 of one row
// read adjacent history windows, so one 512-bit load feeds both and one 512-bit FMA advances
// both accumulators, and each window is loaded once for every accumulator that uses it. That
// halves the history loads and FP instructions of the AVX2 kernel, which is bound by its loads
// rather than its FMAs. Every lane performs the same fused multiply-adds in the same row order
// as the AVX2 kernel, and the epilogue applies the same operations per 256-bit half, so the
// output is bit-identical.

[[nodiscard]] inline __m512 d4x4_duplicate(const __m256 value) noexcept { return _mm512_castpd_ps(_mm512_broadcast_f64x4(_mm256_castps_pd(value))); }

[[nodiscard]] inline __m512 d4x4_xor(const __m512 value, const __m512 mask) noexcept {
    return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(value), _mm512_castps_si512(mask)));
}

/** permute2f128(value, value, 0x01) applied to each 256-bit half. */
[[nodiscard]] inline __m512 d4x4_swap_lanes(const __m512 value) noexcept { return _mm512_shuffle_f32x4(value, value, _MM_SHUFFLE(2, 3, 0, 1)); }

inline void d4x4_butterfly(__m512& even, __m512& odd) noexcept {
    const __m512 left = even;
    const __m512 right = odd;
    even = _mm512_add_ps(left, right);
    odd = _mm512_sub_ps(left, right);
}

[[nodiscard]] inline __m512 d4x4_multiply_complex(const __m512 value, const float real, const float imag) noexcept {
    const __m512 imag_sign = _mm512_setr_ps(-imag, -imag, -imag, -imag, imag, imag, imag, imag, -imag, -imag, -imag, -imag, imag, imag, imag, imag);
    return _mm512_fmadd_ps(d4x4_swap_lanes(value), imag_sign, _mm512_mul_ps(value, _mm512_set1_ps(real)));
}

[[nodiscard]] inline __m512 d4x4_multiply_by_i(const __m512 value) noexcept {
    constexpr int negative = -2147483647 - 1;
    const __m512 sign =
        _mm512_castsi512_ps(_mm512_setr_epi32(negative, negative, negative, negative, 0, 0, 0, 0, negative, negative, negative, negative, 0, 0, 0, 0));
    return d4x4_xor(d4x4_swap_lanes(value), sign);
}

inline void d4x4_transpose_four_hops(const __m512* const input, __m512* const pairs) noexcept {
    const __m512 t0 = _mm512_unpacklo_ps(input[0U], input[1U]);
    const __m512 t1 = _mm512_unpackhi_ps(input[0U], input[1U]);
    const __m512 t2 = _mm512_unpacklo_ps(input[2U], input[3U]);
    const __m512 t3 = _mm512_unpackhi_ps(input[2U], input[3U]);
    pairs[0U] = _mm512_shuffle_ps(t0, t2, 0x44);
    pairs[1U] = _mm512_shuffle_ps(t0, t2, 0xEE);
    pairs[2U] = _mm512_shuffle_ps(t1, t3, 0x44);
    pairs[3U] = _mm512_shuffle_ps(t1, t3, 0xEE);
}

struct D4x4Constants final {
    __m512 rotation_re;
    __m512 rotation_im;
    __m512i emit_lanes;
    __m512 signs_a;
    __m512 signs_b;
};

[[nodiscard]] inline D4x4Constants d4x4_constants(const PfbChannelizerData& data, const std::size_t post_phase) noexcept {
    // The accumulators are not lane-reversed, so they take the branch rotation reversed.
    const __m256i reverse = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const auto& phase = d4x4_phase_maps[post_phase];
    const __m256i lanes = _mm256_load_si256(reinterpret_cast<const __m256i*>(phase.lanes.data()));
    return {.rotation_re = d4x4_duplicate(_mm256_permutevar8x32_ps(_mm256_load_ps(PfbChannelizerAccess::branch_rotation_re(data)), reverse)),
            .rotation_im = d4x4_duplicate(_mm256_permutevar8x32_ps(_mm256_load_ps(PfbChannelizerAccess::branch_rotation_im(data)), reverse)),
            .emit_lanes = _mm512_inserti64x4(_mm512_castsi256_si512(lanes), _mm256_add_epi32(lanes, _mm256_set1_epi32(8)), 1),
            .signs_a = d4x4_duplicate(_mm256_load_ps(phase.signs_a.data())),
            .signs_b = d4x4_duplicate(_mm256_load_ps(phase.signs_b.data()))};
}

/**
 * Accumulate I and Q of eight hops. Pair p covers hops (0, 2), (1, 3), (4, 6), (5, 7); its row-r
 * window starts 4 * (first hop) - 8 * r samples after the origin and the second hop's window
 * follows it directly. Windows are visited from the newest backwards so each pair receives its
 * rows in ascending order, exactly as the AVX2 kernel accumulates them.
 */
template <std::size_t FixedRows>
UNI_SIMD_D4X4_INLINE void d4x4_accumulate(const float* const coefficients, const std::size_t rows, const float* window_i, const float* window_q,
                                          __m512 (&accumulator_i)[4U], __m512 (&accumulator_q)[4U]) noexcept {
    constexpr std::size_t bins_count = 8U;
    __m512 samples_i;
    __m512 samples_q;
    const auto load = [&](const std::ptrdiff_t quarter) noexcept {
        samples_i = _mm512_loadu_ps(window_i + 4 * quarter);
        samples_q = _mm512_loadu_ps(window_q + 4 * quarter);
    };
    const auto advance = [&](const std::size_t samples) noexcept {
        window_i -= samples;
        window_q -= samples;
    };
    const auto tap = [&](const std::size_t pair, const __m512 coefficient) noexcept {
        accumulator_i[pair] = _mm512_fmadd_ps(samples_i, coefficient, accumulator_i[pair]);
        accumulator_q[pair] = _mm512_fmadd_ps(samples_q, coefficient, accumulator_q[pair]);
    };
    const auto coefficient = [&](const std::size_t row) noexcept { return d4x4_duplicate(_mm256_load_ps(coefficients + row * bins_count)); };

    std::size_t row = 0U;
    for (; row + 4U <= rows; row += 4U, advance(4U * bins_count)) {
        const __m512 c0 = coefficient(row);
        const __m512 c1 = coefficient(row + 1U);
        const __m512 c2 = coefficient(row + 2U);
        const __m512 c3 = coefficient(row + 3U);
        load(5);
        tap(3U, c0);
        load(4);
        tap(2U, c0);
        load(3);
        tap(3U, c1);
        load(2);
        tap(2U, c1);
        load(1);
        tap(1U, c0);
        tap(3U, c2);
        load(0);
        tap(0U, c0);
        tap(2U, c2);
        load(-1);
        tap(1U, c1);
        tap(3U, c3);
        load(-2);
        tap(0U, c1);
        tap(2U, c3);
        load(-3);
        tap(1U, c2);
        load(-4);
        tap(0U, c2);
        load(-5);
        tap(1U, c3);
        load(-6);
        tap(0U, c3);
    }
    if (FixedRows == 0U || FixedRows % 4U >= 2U) {
        for (; row + 2U <= rows; row += 2U, advance(2U * bins_count)) {
            const __m512 c0 = coefficient(row);
            const __m512 c1 = coefficient(row + 1U);
            load(5);
            tap(3U, c0);
            load(4);
            tap(2U, c0);
            load(3);
            tap(3U, c1);
            load(2);
            tap(2U, c1);
            load(1);
            tap(1U, c0);
            load(0);
            tap(0U, c0);
            load(-1);
            tap(1U, c1);
            load(-2);
            tap(0U, c1);
        }
    }
    if (FixedRows == 0U || FixedRows % 2U != 0U) {
        for (; row < rows; ++row, advance(bins_count)) {
            const __m512 c0 = coefficient(row);
            load(5);
            tap(3U, c0);
            load(4);
            tap(2U, c0);
            load(1);
            tap(1U, c0);
            load(0);
            tap(0U, c0);
        }
    }
}

/** Eight hops starting at ring position `batch_cursor`; both 256-bit halves then run the AVX2 epilogue. */
template <bool Aligned, std::size_t FixedRows>
UNI_SIMD_D4X4_INLINE void d4x4_batch(const PfbChannelizerData& data, const PfbChannelizerBlock& block, const D4x4Constants& constants,
                                     const std::size_t batch_cursor, const std::size_t output_index) noexcept {
    constexpr std::size_t bins_count = 8U;
    const std::size_t rows = FixedRows == 0U ? PfbChannelizerAccess::rows(data) : FixedRows;
    const std::size_t history_size = PfbChannelizerAccess::history_size(data);

    // Row 0 of hop 0 ends at the batch's first sample. Both copies of the ring hold the same
    // samples; when the last hop would run past the upper copy the lower one serves the same
    // windows, and the ring is sized so the oldest row still fits.
    std::size_t origin = batch_cursor + history_size - (bins_count - 1U);
    if (batch_cursor + 7U * 4U + 1U > history_size) {
        origin -= history_size;
    }
    __m512 pairs_re[4U]{_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
    __m512 pairs_im[4U]{_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
    d4x4_accumulate<FixedRows>(PfbChannelizerAccess::reversed_coefficients(data), rows, PfbChannelizerAccess::history_i(data) + origin,
                               PfbChannelizerAccess::history_q(data) + origin, pairs_re, pairs_im);

    // Regroup to hop k in the low half and hop k + 4 in the high half.
    const auto regroup = [](const __m512* const pairs, __m512* const hops) noexcept {
        hops[0U] = _mm512_shuffle_f32x4(pairs[0U], pairs[2U], 0x44);
        hops[1U] = _mm512_shuffle_f32x4(pairs[1U], pairs[3U], 0x44);
        hops[2U] = _mm512_shuffle_f32x4(pairs[0U], pairs[2U], 0xEE);
        hops[3U] = _mm512_shuffle_f32x4(pairs[1U], pairs[3U], 0xEE);
    };
    __m512 hops_re[4U];
    __m512 hops_im[4U];
    regroup(pairs_re, hops_re);
    regroup(pairs_im, hops_im);

    __m512 rotated_re[4U];
    __m512 rotated_im[4U];
    for (std::size_t hop = 0U; hop < 4U; ++hop) {
        rotated_re[hop] = _mm512_fmsub_ps(hops_re[hop], constants.rotation_re, _mm512_mul_ps(hops_im[hop], constants.rotation_im));
        rotated_im[hop] = _mm512_fmadd_ps(hops_re[hop], constants.rotation_im, _mm512_mul_ps(hops_im[hop], constants.rotation_re));
    }
    __m512 real_pairs[4U];
    __m512 imag_pairs[4U];
    d4x4_transpose_four_hops(rotated_re, real_pairs);
    d4x4_transpose_four_hops(rotated_im, imag_pairs);

    // real_pairs[k] carries lanes k and k + 4 of each half, which hold branches 7 - k and 3 - k.
    const __m512i low_lanes = _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27);
    const __m512i high_lanes = _mm512_setr_epi32(4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31);
    __m512 bins[8U];
    for (std::size_t index = 0U; index < 4U; ++index) {
        bins[7U - index] = _mm512_permutex2var_ps(real_pairs[index], low_lanes, imag_pairs[index]);
        bins[3U - index] = _mm512_permutex2var_ps(real_pairs[index], high_lanes, imag_pairs[index]);
    }

    constexpr float root_half = 0.70710678118654752440f;
    for (std::size_t index = 0U; index < 4U; ++index) {
        d4x4_butterfly(bins[index], bins[index + 4U]);
    }
    bins[5U] = d4x4_multiply_complex(bins[5U], root_half, root_half);
    bins[6U] = d4x4_multiply_by_i(bins[6U]);
    bins[7U] = d4x4_multiply_complex(bins[7U], -root_half, root_half);

    d4x4_butterfly(bins[0U], bins[2U]);
    d4x4_butterfly(bins[1U], bins[3U]);
    bins[3U] = d4x4_multiply_by_i(bins[3U]);
    d4x4_butterfly(bins[4U], bins[6U]);
    d4x4_butterfly(bins[5U], bins[7U]);
    bins[7U] = d4x4_multiply_by_i(bins[7U]);

    const auto emit = [&](const std::size_t output, const __m512 value, const __m512 signs) noexcept {
        const __m512 packed = d4x4_xor(_mm512_permutexvar_ps(constants.emit_lanes, value), signs);
        float* const destination = block.outputs[output].data() + 2U * output_index;
        const __m256 first = _mm512_castps512_ps256(packed);
        const __m256 second = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(packed), 1));
        if constexpr (Aligned) {
            _mm256_store_ps(destination, first);
            _mm256_store_ps(destination + 8U, second);
        } else {
            _mm256_storeu_ps(destination, first);
            _mm256_storeu_ps(destination + 8U, second);
        }
    };
    emit(0U, _mm512_sub_ps(bins[2U], bins[3U]), constants.signs_a);
    emit(1U, _mm512_sub_ps(bins[6U], bins[7U]), constants.signs_b);
    emit(2U, _mm512_add_ps(bins[0U], bins[1U]), constants.signs_a);
    emit(3U, _mm512_add_ps(bins[4U], bins[5U]), constants.signs_b);
}

template <bool Aligned, std::size_t FixedRows> void d4x4_runs(PfbChannelizerData& data, const PfbChannelizerBlock& block, PfbD4x4RunState& state) noexcept {
    constexpr std::size_t group_samples = 16U;
    static_assert(pfb_d4x4_runs_span % (2U * group_samples) == 0U);
    const std::size_t history_size = PfbChannelizerAccess::history_size(data);
    const std::size_t history_mask = history_size - 1U;
    float* const history_i = PfbChannelizerAccess::history_i(data);
    float* const history_q = PfbChannelizerAccess::history_q(data);
    const std::size_t input_count = block.input.size() / 2U;
    const float* const input = block.input.data();
    const D4x4Constants constants = d4x4_constants(data, state.post_phase);
    std::size_t cursor = state.cursor;
    std::size_t input_index = state.input_index;
    std::size_t produced = state.produced;

    const auto write_group = [&]() noexcept {
        if (cursor + group_samples <= history_size) {
            const float* const source = input + 2U * input_index;
            for (std::size_t half = 0U; half < 2U; ++half) {
                const __m256 first = _mm256_loadu_ps(source + 16U * half);
                const __m256 second = _mm256_loadu_ps(source + 16U * half + 8U);
                const __m256 real = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(_mm256_shuffle_ps(first, second, 0x88)), 0xD8));
                const __m256 imag = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(_mm256_shuffle_ps(first, second, 0xDD)), 0xD8));
                const std::size_t target = cursor + 8U * half;
                _mm256_storeu_ps(history_i + target, real);
                _mm256_storeu_ps(history_i + target + history_size, real);
                _mm256_storeu_ps(history_q + target, imag);
                _mm256_storeu_ps(history_q + target + history_size, imag);
            }
        } else {
            for (std::size_t offset = 0U; offset < group_samples; ++offset) {
                const std::size_t target = (cursor + offset) & history_mask;
                const float sample_re = input[2U * (input_index + offset)];
                const float sample_im = input[2U * (input_index + offset) + 1U];
                history_i[target] = sample_re;
                history_i[target + history_size] = sample_re;
                history_q[target] = sample_im;
                history_q[target + history_size] = sample_im;
            }
        }
        input_index += group_samples;
        cursor = (cursor + group_samples) & history_mask;
    };

    // Stage a span into the ring, then filter it.
    while (input_count - input_index >= pfb_d4x4_runs_span) {
        const std::size_t lookahead_cursor = cursor;
        for (std::size_t group = 0U; group < pfb_d4x4_runs_span / group_samples; ++group) {
            write_group();
        }
        for (std::size_t group = 0U; group < pfb_d4x4_runs_span / group_samples; group += 2U) {
            d4x4_batch<Aligned, FixedRows>(data, block, constants, (lookahead_cursor + group * group_samples) & history_mask, produced);
            produced += 8U;
        }
    }
    state.cursor = cursor;
    state.input_index = input_index;
    state.produced = produced;
}
} // namespace

void PfbD4x4Runs_avx512(PfbChannelizerData& data, const PfbChannelizerBlock& block, PfbD4x4RunState& state) noexcept {
    bool aligned = true;
    for (std::size_t output = 0U; output < 4U; ++output) {
        aligned = aligned && (reinterpret_cast<std::uintptr_t>(block.outputs[output].data()) & 31U) == 0U;
    }
    if (aligned && PfbChannelizerAccess::rows(data) == 22U) {
        d4x4_runs<true, 22U>(data, block, state);
    } else if (aligned) {
        d4x4_runs<true, 0U>(data, block, state);
    } else {
        d4x4_runs<false, 0U>(data, block, state);
    }
}

bool PfbChannelizer_supports_avx512(const PfbChannelizerData& data) noexcept {
    return data.bin_count() == 32U ||
           (data.bin_count() == 8U && data.selected_output_count() > 1U);
}

std::size_t PfbChannelizer_avx512(PfbChannelizerData& data,
                                  const PfbChannelizerBlock& block) noexcept {
    switch (data.bin_count()) {
    case 8U:
        return data.grid_offset() == PfbGridOffset::half_bins
                   ? process_8<true>(data, block)
                   : process_8<false>(data, block);
    case 16U:
        return process_selected<16U>(data, block);
    case 32U:
        return process_selected<32U>(data, block);
    default:
        return PfbChannelizer_avx2fma(data, block);
    }
}

#else

void PfbD4x4Runs_avx512(PfbChannelizerData&, const PfbChannelizerBlock&, PfbD4x4RunState&) noexcept {}

bool PfbChannelizer_supports_avx512(const PfbChannelizerData&) noexcept {
    return false;
}

std::size_t PfbChannelizer_avx512(PfbChannelizerData& data,
                                  const PfbChannelizerBlock& block) noexcept {
    return PfbChannelizer_generic(data, block);
}

#endif

} // namespace uni::simd::detail
