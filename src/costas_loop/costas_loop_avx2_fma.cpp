#include "costas_loop_internal.hpp"

#if UNI_SIMD_HAVE_AVX2_FMA

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace uni::simd::kernels {

namespace {

/**
 * Load one complex sample from each of the four channels and transpose the interleaved
 * pairs into a real vector and an imaginary vector.
 *
 * Each channel contributes a 64-bit `[re, im]` pair that `movlps`/`movhps` load straight into
 * a vector half, so two loads per vector and one shuffle per component are enough. `__m64`
 * is a may-alias type, which keeps the float buffers legal to access this way and avoids
 * bouncing the pairs through general-purpose registers and the stack.
 */
struct Deinterleaved final {
    __m128 real;
    __m128 imag;
};

struct Channels final {
    float* lane0;
    float* lane1;
    float* lane2;
    float* lane3;
};

[[nodiscard]] inline Deinterleaved LoadChannels(const Channels& channels, const std::size_t sample) noexcept {
    const std::size_t index = sample * 2U;
    const __m128 low = _mm_loadh_pi(_mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(channels.lane0 + index)),
                                    reinterpret_cast<const __m64*>(channels.lane1 + index)); // [re0, im0, re1, im1]
    const __m128 high = _mm_loadh_pi(_mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(channels.lane2 + index)),
                                     reinterpret_cast<const __m64*>(channels.lane3 + index)); // [re2, im2, re3, im3]
    return {_mm_shuffle_ps(low, high, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_ps(low, high, _MM_SHUFFLE(3, 1, 3, 1))};
}

inline void StoreChannels(const Channels& channels, const std::size_t sample, const __m128 real, const __m128 imag) noexcept {
    const std::size_t index = sample * 2U;
    const __m128 low = _mm_unpacklo_ps(real, imag);  // [re0, im0, re1, im1]
    const __m128 high = _mm_unpackhi_ps(real, imag); // [re2, im2, re3, im3]
    _mm_storel_pi(reinterpret_cast<__m64*>(channels.lane0 + index), low);
    _mm_storeh_pi(reinterpret_cast<__m64*>(channels.lane1 + index), low);
    _mm_storel_pi(reinterpret_cast<__m64*>(channels.lane2 + index), high);
    _mm_storeh_pi(reinterpret_cast<__m64*>(channels.lane3 + index), high);
}

} // namespace

void QpskCostas4_avx2(uni_simd_qpsk_costas4_t& kernel, const uni_simd_qpsk_costas4_block_t& block) noexcept {
    __m128 phase = _mm_loadu_ps(kernel.state.phase);
    __m128 phase_cos = _mm_loadu_ps(kernel.state.phase_cos);
    __m128 phase_sin = _mm_loadu_ps(kernel.state.phase_sin);
    __m128 frequency = _mm_loadu_ps(kernel.state.frequency);
    __m128 last_error = _mm_loadu_ps(kernel.state.last_error);
    // Steps decided but not yet applied (see UNI_SIMD_QPSK_COSTAS4_LOOP_DELAY), as a ring in
    // L1: slot `pending_slot` holds the oldest step, which is read and then overwritten by the
    // newest. A step is stored `delay` samples before it is loaded, so the store-to-load
    // round trip stays off the recurrence, while holding all of them in registers would not
    // leave enough registers for the loop.
    constexpr std::size_t delay = UNI_SIMD_QPSK_COSTAS4_LOOP_DELAY;
    static_assert((delay & (delay - 1U)) == 0U, "the pending ring is indexed with a mask");
    struct PendingStep final {
        __m128 step;
        __m128 cos;
        __m128 sin;
    };
    PendingStep pending[delay];
    for (std::size_t index = 0U; index < delay; ++index) {
        pending[index] = {_mm_loadu_ps(kernel.state.pending_step[index]), _mm_loadu_ps(kernel.state.pending_step_cos[index]),
                          _mm_loadu_ps(kernel.state.pending_step_sin[index])};
    }
    std::size_t pending_slot = 0U;
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
    // Held in locals so the output stores cannot be assumed to alias the pointer table,
    // which would force a reload of all four pointers for every sample.
    const Channels channels{block.channels[0], block.channels[1], block.channels[2], block.channels[3]};

    constexpr std::size_t renormalization_period = 512U;
    for (std::size_t chunk_begin = 0U; chunk_begin < block.sample_count;) {
        const std::size_t until_normalization = renormalization_period - kernel.samples_since_normalization;
        const std::size_t chunk_end = std::min(chunk_begin + until_normalization, block.sample_count);
        for (std::size_t sample = chunk_begin; sample < chunk_end; ++sample) {
            const Deinterleaved loaded = LoadChannels(channels, sample);
            const __m128 real = unit_gain ? loaded.real : _mm_mul_ps(loaded.real, input_gain);
            const __m128 imag = unit_gain ? loaded.imag : _mm_mul_ps(loaded.imag, input_gain);
            const __m128 output_real = _mm_fmadd_ps(real, phase_cos, _mm_mul_ps(imag, phase_sin));
            const __m128 output_imag = _mm_fmsub_ps(imag, phase_cos, _mm_mul_ps(real, phase_sin));
            StoreChannels(channels, sample, output_real, output_imag);

            // copysign(1, x) * y is exactly y with x's sign bit applied, so the sign transfer
            // replaces two multiplies on the serial chain with two logical ops.
            __m128 error = _mm_sub_ps(_mm_xor_ps(output_imag, _mm_and_ps(output_real, sign_mask)), _mm_xor_ps(output_real, _mm_and_ps(output_imag, sign_mask)));
            __m128 next_frequency = _mm_fmadd_ps(beta, error, frequency);

            // The error clip and the frequency limit are identities unless a lane leaves its bound,
            // which a locked loop practically never does. Checking the bounds on a predicted
            // branch keeps the four min/max operations off the serial chain; the slow path
            // replays the exact clamped sequence, so results stay bit-identical.
            const __m128 error_outside = _mm_cmp_ps(_mm_and_ps(error, absolute_mask), clip_bound, _CMP_NLE_UQ);
            const __m128 frequency_outside = _mm_cmp_ps(_mm_and_ps(next_frequency, absolute_mask), limit_bound, _CMP_NLE_UQ);
            if (_mm_movemask_ps(_mm_or_ps(error_outside, frequency_outside)) != 0) [[unlikely]] {
                error = _mm_min_ps(_mm_max_ps(error, negative_clip_bound), clip_bound);
                next_frequency = _mm_min_ps(_mm_max_ps(_mm_fmadd_ps(beta, error, frequency), negative_limit_bound), limit_bound);
            }
            last_error = error;
            frequency = next_frequency;
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
            // Advance the phasor by the oldest pending step and queue this sample's step. The
            // phasor only waits for a step decided `delay` samples ago, so the long
            // error -> step chain of one sample runs alongside the rotations of the next ones.
            const PendingStep oldest = pending[pending_slot];
            const __m128 previous_cos = phase_cos;
            phase_cos = _mm_fmsub_ps(previous_cos, oldest.cos, _mm_mul_ps(phase_sin, oldest.sin));
            phase_sin = _mm_fmadd_ps(phase_sin, oldest.cos, _mm_mul_ps(previous_cos, oldest.sin));
            phase = _mm_add_ps(phase, oldest.step);
            pending[pending_slot] = {delta, delta_cos, delta_sin};
            pending_slot = (pending_slot + 1U) & (delay - 1U);
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
    // Back to oldest-first order.
    for (std::size_t index = 0U; index < delay; ++index) {
        const PendingStep& step = pending[(pending_slot + index) & (delay - 1U)];
        _mm_storeu_ps(kernel.state.pending_step[index], step.step);
        _mm_storeu_ps(kernel.state.pending_step_cos[index], step.cos);
        _mm_storeu_ps(kernel.state.pending_step_sin[index], step.sin);
    }
}

} // namespace uni::simd::kernels

#endif
