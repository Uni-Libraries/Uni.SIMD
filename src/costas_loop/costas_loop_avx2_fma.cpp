#include "costas_loop_internal.hpp"

#if UNI_SIMD_HAVE_AVX2_FMA

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace uni::simd::kernels {

namespace {

/**
 * Load one complex sample from each of the four channels and transpose the interleaved
 * pairs into a real vector and an imaginary vector.
 *
 * Each channel contributes a 64-bit `[re, im]` pair, so two unpack steps and one
 * `movelh`/`movehl` pair are enough; this avoids the four-way scalar gather that the
 * previous revision performed for every component of every sample.
 */
struct Deinterleaved final {
    __m128 real;
    __m128 imag;
};

[[nodiscard]] inline Deinterleaved LoadChannels(float* const* const channels, const std::size_t sample) noexcept {
    const std::size_t index = sample * 2U;
    std::uint64_t bits[4];
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
        std::memcpy(&bits[lane], channels[lane] + index, sizeof(bits[lane]));
    }
    const __m128 pair0 = _mm_castsi128_ps(_mm_cvtsi64_si128(static_cast<long long>(bits[0])));
    const __m128 pair1 = _mm_castsi128_ps(_mm_cvtsi64_si128(static_cast<long long>(bits[1])));
    const __m128 pair2 = _mm_castsi128_ps(_mm_cvtsi64_si128(static_cast<long long>(bits[2])));
    const __m128 pair3 = _mm_castsi128_ps(_mm_cvtsi64_si128(static_cast<long long>(bits[3])));
    const __m128 low = _mm_unpacklo_ps(pair0, pair1);  // [re0, re1, im0, im1]
    const __m128 high = _mm_unpacklo_ps(pair2, pair3); // [re2, re3, im2, im3]
    return {_mm_movelh_ps(low, high), _mm_movehl_ps(high, low)};
}

inline void StoreChannels(float* const* const channels, const std::size_t sample, const __m128 real, const __m128 imag) noexcept {
    const std::size_t index = sample * 2U;
    const __m128 low = _mm_unpacklo_ps(real, imag);  // [re0, im0, re1, im1]
    const __m128 high = _mm_unpackhi_ps(real, imag); // [re2, im2, re3, im3]
    const std::uint64_t bits[4]{
        static_cast<std::uint64_t>(_mm_cvtsi128_si64(_mm_castps_si128(low))),
        static_cast<std::uint64_t>(_mm_cvtsi128_si64(_mm_castps_si128(_mm_movehl_ps(low, low)))),
        static_cast<std::uint64_t>(_mm_cvtsi128_si64(_mm_castps_si128(high))),
        static_cast<std::uint64_t>(_mm_cvtsi128_si64(_mm_castps_si128(_mm_movehl_ps(high, high)))),
    };
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
        std::memcpy(channels[lane] + index, &bits[lane], sizeof(bits[lane]));
    }
}

} // namespace

void QpskCostas4_avx2(uni_simd_qpsk_costas4_t& kernel, const uni_simd_qpsk_costas4_block_t& block) noexcept {
    __m128 phase = _mm_loadu_ps(kernel.state.phase);
    __m128 phase_cos = _mm_loadu_ps(kernel.state.phase_cos);
    __m128 phase_sin = _mm_loadu_ps(kernel.state.phase_sin);
    __m128 frequency = _mm_loadu_ps(kernel.state.frequency);
    __m128 last_error = _mm_loadu_ps(kernel.state.last_error);
    const __m128 alpha = _mm_loadu_ps(kernel.config.alpha);
    const __m128 beta = _mm_loadu_ps(kernel.config.beta);
    const __m128 error_clip = _mm_loadu_ps(kernel.config.error_clip);
    const __m128 frequency_limit = _mm_loadu_ps(block.frequency_limit);
    const __m128 input_gain = _mm_loadu_ps(block.input_gain);
    const __m128 zero = _mm_setzero_ps();
    const __m128 one = _mm_set1_ps(1.0f);
    const __m128 sign_mask = _mm_set1_ps(-0.0f);
    const __m128 absolute_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    const __m128 approximation_limit = _mm_set1_ps(0.25f);
    const __m128 one_over_120 = _mm_set1_ps(1.0f / 120.0f);
    const __m128 minus_one_over_6 = _mm_set1_ps(-1.0f / 6.0f);
    const __m128 one_over_24 = _mm_set1_ps(1.0f / 24.0f);
    const __m128 minus_half = _mm_set1_ps(-0.5f);
    // Clipping and limiting are loop-invariant per-lane configuration. Disabled lanes get
    // an infinite bound instead of a per-sample select, which keeps min/max exact and takes
    // the blend off the serial phase-update chain.
    const __m128 infinity = _mm_set1_ps(std::numeric_limits<float>::infinity());
    const __m128 clip_bound = _mm_blendv_ps(infinity, error_clip, _mm_cmpgt_ps(error_clip, zero));
    const __m128 limit_bound = _mm_blendv_ps(infinity, frequency_limit, _mm_cmpgt_ps(frequency_limit, zero));
    const __m128 negative_clip_bound = _mm_sub_ps(zero, clip_bound);
    const __m128 negative_limit_bound = _mm_sub_ps(zero, limit_bound);
    const bool unit_gain = _mm_movemask_ps(_mm_cmpeq_ps(input_gain, one)) == 0x0f;

    constexpr std::size_t renormalization_period = 512U;
    for (std::size_t chunk_begin = 0U; chunk_begin < block.sample_count;) {
        const std::size_t until_normalization = renormalization_period - kernel.samples_since_normalization;
        const std::size_t chunk_end = std::min(chunk_begin + until_normalization, block.sample_count);
        for (std::size_t sample = chunk_begin; sample < chunk_end; ++sample) {
            const Deinterleaved loaded = LoadChannels(block.channels, sample);
            const __m128 real = unit_gain ? loaded.real : _mm_mul_ps(loaded.real, input_gain);
            const __m128 imag = unit_gain ? loaded.imag : _mm_mul_ps(loaded.imag, input_gain);
            const __m128 output_real = _mm_fmadd_ps(real, phase_cos, _mm_mul_ps(imag, phase_sin));
            const __m128 output_imag = _mm_fmsub_ps(imag, phase_cos, _mm_mul_ps(real, phase_sin));
            StoreChannels(block.channels, sample, output_real, output_imag);

            // copysign(1, x) * y is exactly y with x's sign bit applied, so the sign transfer
            // replaces two multiplies on the serial chain with two logical ops.
            __m128 error = _mm_sub_ps(_mm_xor_ps(output_imag, _mm_and_ps(output_real, sign_mask)), _mm_xor_ps(output_real, _mm_and_ps(output_imag, sign_mask)));
            error = _mm_min_ps(_mm_max_ps(error, negative_clip_bound), clip_bound);
            last_error = error;

            frequency = _mm_min_ps(_mm_max_ps(_mm_fmadd_ps(beta, error, frequency), negative_limit_bound), limit_bound);
            const __m128 delta = _mm_fmadd_ps(alpha, error, frequency);

            const __m128 squared = _mm_mul_ps(delta, delta);
            const __m128 delta_squared = _mm_mul_ps(delta, squared);
            const __m128 sin_polynomial = _mm_fmadd_ps(squared, one_over_120, minus_one_over_6);
            const __m128 cos_polynomial = _mm_fmadd_ps(squared, _mm_fmadd_ps(squared, one_over_24, minus_half), one);
            __m128 delta_sin = _mm_fmadd_ps(delta_squared, sin_polynomial, delta);
            __m128 delta_cos = cos_polynomial;
            const int exceptional_mask = _mm_movemask_ps(_mm_cmpgt_ps(_mm_and_ps(delta, absolute_mask), approximation_limit));
            if (exceptional_mask != 0) [[unlikely]] {
                alignas(16) float delta_lanes[4];
                alignas(16) float delta_sin_lanes[4];
                alignas(16) float delta_cos_lanes[4];
                _mm_store_ps(delta_lanes, delta);
                _mm_store_ps(delta_sin_lanes, delta_sin);
                _mm_store_ps(delta_cos_lanes, delta_cos);
                for (std::size_t lane = 0U; lane < 4U; ++lane) {
                    if ((exceptional_mask & (1 << lane)) != 0) {
                        Costas4SinCos(delta_lanes[lane], delta_sin_lanes[lane], delta_cos_lanes[lane]);
                    }
                }
                delta_sin = _mm_load_ps(delta_sin_lanes);
                delta_cos = _mm_load_ps(delta_cos_lanes);
            }
            const __m128 previous_cos = phase_cos;
            phase_cos = _mm_fmsub_ps(previous_cos, delta_cos, _mm_mul_ps(phase_sin, delta_sin));
            phase_sin = _mm_fmadd_ps(phase_sin, delta_cos, _mm_mul_ps(previous_cos, delta_sin));
            phase = _mm_add_ps(phase, delta);
        }
        kernel.samples_since_normalization += chunk_end - chunk_begin;
        chunk_begin = chunk_end;
        if (kernel.samples_since_normalization == renormalization_period) {
            kernel.samples_since_normalization = 0U;
            _mm_storeu_ps(kernel.state.phase, phase);
            _mm_storeu_ps(kernel.state.phase_cos, phase_cos);
            _mm_storeu_ps(kernel.state.phase_sin, phase_sin);
            for (std::size_t lane = 0U; lane < 4U; ++lane) {
                Costas4Normalize(kernel.state.phase[lane], kernel.state.phase_cos[lane], kernel.state.phase_sin[lane]);
            }
            phase = _mm_loadu_ps(kernel.state.phase);
            phase_cos = _mm_loadu_ps(kernel.state.phase_cos);
            phase_sin = _mm_loadu_ps(kernel.state.phase_sin);
        }
    }
    _mm_storeu_ps(kernel.state.phase, phase);
    _mm_storeu_ps(kernel.state.phase_cos, phase_cos);
    _mm_storeu_ps(kernel.state.phase_sin, phase_sin);
    _mm_storeu_ps(kernel.state.frequency, frequency);
    _mm_storeu_ps(kernel.state.last_error, last_error);
}

} // namespace uni::simd::kernels

#endif
