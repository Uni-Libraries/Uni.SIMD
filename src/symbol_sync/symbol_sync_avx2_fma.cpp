#include "symbol_sync_internal.hpp"

#if UNI_SIMD_HAVE_AVX2_FMA

#include <immintrin.h>

#include <algorithm>
#include <cstdint>

namespace uni::simd::kernels {

namespace {

/**
 * Layout: lane k is channel k. Loop state (time, omega, ...) lives in the four doubles of a
 * __m256d, complex values as separate real and imaginary __m128 (structure of arrays), so the
 * detector error is plain lane-wise arithmetic.
 *
 * Each timing loop is a serial recurrence: the position of the next symbol depends on the
 * detector error at the current one, which needs the interpolated samples there. That chain of
 * latencies bounds a step, and running the four independent channels in the lanes of one
 * vector yields four symbols per step instead of one. Everything here is arranged to keep the
 * chain short:
 *   - the sample index is truncated from the position while mu comes from floor() in parallel,
 *     and the interpolation weights only depend on mu, so they are ready before the samples;
 *   - the errors are formed lane-wise, without horizontal sums;
 *   - between omega clamps the next position and the next Gardner mid-point are each one
 *     multiply-add after the error (see advance()).
 * The common case runs in batches that also leave out work hanging off the chain (bounds
 * clamping, per-lane masks, the detector not in use, and the lock tracking of the automatic
 * detector switch, which is replayed after the batch), so consecutive steps overlap.
 */

/** Steps of one batch; also the length of the error log replayed by the lock tracking. */
constexpr std::size_t kBatch = 64U;

struct Complex4 final {
    __m128 re;
    __m128 im;
};

[[nodiscard]] inline __m256d LaneMask(const int bits) noexcept {
    const __m256i select = _mm256_setr_epi64x(1, 2, 4, 8);
    return _mm256_castsi256_pd(_mm256_cmpeq_epi64(_mm256_and_si256(_mm256_set1_epi64x(bits), select), select));
}

[[nodiscard]] inline __m128 LaneMask4(const int bits) noexcept {
    const __m128i select = _mm_setr_epi32(1, 2, 4, 8);
    return _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_and_si128(_mm_set1_epi32(bits), select), select));
}

[[nodiscard]] inline int LaneBits(const std::uint32_t (&flags)[kSymbolSync4Lanes]) noexcept {
    return (flags[0] != 0U ? 1 : 0) | (flags[1] != 0U ? 2 : 0) | (flags[2] != 0U ? 4 : 0) | (flags[3] != 0U ? 8 : 0);
}

[[nodiscard]] inline double HorizontalMax(const __m256d value) noexcept {
    const __m128d pairs = _mm_max_pd(_mm256_castpd256_pd128(value), _mm256_extractf128_pd(value, 1));
    return _mm_cvtsd_f64(_mm_max_sd(pairs, _mm_unpackhi_pd(pairs, pairs)));
}

/** sign(decision) * value per float: value with its sign flipped where decision is negative (QPSK slicer times value). */
[[nodiscard]] inline __m128 SignTimes(const __m128 decision, const __m128 value) noexcept {
    return _mm_xor_ps(value, _mm_and_ps(_mm_set1_ps(-0.0f), decision));
}

/** Gardner: Re{conj(mid) * (previous - current)}. */
[[nodiscard]] inline __m128 GardnerError(const Complex4& previous, const Complex4& mid, const Complex4& current) noexcept {
    return _mm_add_ps(_mm_mul_ps(mid.re, _mm_sub_ps(previous.re, current.re)), _mm_mul_ps(mid.im, _mm_sub_ps(previous.im, current.im)));
}

/** Mueller and Muller with QPSK decisions: Re{conj(d_previous) * current - conj(d_current) * previous}. */
[[nodiscard]] inline __m128 MuellerMullerError(const Complex4& previous, const Complex4& current) noexcept {
    return _mm_add_ps(_mm_sub_ps(SignTimes(previous.re, current.re), SignTimes(current.re, previous.re)),
                      _mm_sub_ps(SignTimes(previous.im, current.im), SignTimes(current.im, previous.im)));
}

/** Power of the samples the detector used. */
[[nodiscard]] inline __m128 DetectorEnergy(const Complex4& previous, const Complex4& mid, const Complex4& current, const bool mueller_muller) noexcept {
    __m128 power_re = _mm_fmadd_ps(previous.re, previous.re, _mm_mul_ps(current.re, current.re));
    __m128 power_im = _mm_fmadd_ps(previous.im, previous.im, _mm_mul_ps(current.im, current.im));
    if (!mueller_muller) {
        power_re = _mm_fmadd_ps(mid.re, mid.re, power_re);
        power_im = _mm_fmadd_ps(mid.im, mid.im, power_im);
    }
    return _mm_add_ps(power_re, power_im);
}

/** Interleaved {re, im} pairs of four lanes -> real and imaginary vectors. */
[[nodiscard]] inline Complex4 Deinterleave(const float* const pairs) noexcept {
    const __m128 low = _mm_loadu_ps(pairs);
    const __m128 high = _mm_loadu_ps(pairs + 4);
    return {_mm_shuffle_ps(low, high, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_ps(low, high, _MM_SHUFFLE(3, 1, 3, 1))};
}

inline void Interleave(const Complex4& value, float* const pairs) noexcept {
    _mm_storeu_ps(pairs, _mm_unpacklo_ps(value.re, value.im));
    _mm_storeu_ps(pairs + 4, _mm_unpackhi_ps(value.re, value.im));
}

[[nodiscard]] inline Complex4 Select(const Complex4& unchanged, const Complex4& changed, const __m128 mask) noexcept {
    return {_mm_blendv_ps(unchanged.re, changed.re, mask), _mm_blendv_ps(unchanged.im, changed.im, mask)};
}

struct Inputs final {
    const float* data[kSymbolSync4Lanes];
    /** length - 3 per lane: the largest `index` whose stencil [index - 1, index + 2] fits. */
    __m128i last_index;
};

/**
 * Interpolates every lane at its `position`.
 *
 * The stencil of each lane (four complex samples = 32 bytes) is one unaligned load; a 4x8
 * transpose then gives the real and imaginary parts of tap j of every lane, with taps 0 and 2
 * (1 and 3) sharing a register, so the weighted sums are one multiply and one multiply-add per
 * component plus a final fold of the two halves. With `Clamp` the stencil is clamped into the
 * data, for idle lanes of a masked step or a corrupted time; batches check the bounds in their
 * loop condition instead.
 */
template <bool Cubic, bool Clamp>
[[nodiscard]] inline Complex4 Interpolate(const Inputs& inputs, const __m256d position) noexcept {
    __m128i index = _mm256_cvttpd_epi32(position);
    if constexpr (Clamp) {
        index = _mm_min_epi32(_mm_max_epi32(index, _mm_set1_epi32(1)), inputs.last_index);
    }
    const __m128 mu = _mm256_cvtpd_ps(_mm256_sub_pd(position, _mm256_round_pd(position, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)));
    const auto load = [&](const std::size_t lane, const int sample) noexcept {
        // The stencil starts one sample before `sample`.
        return _mm256_loadu_ps(inputs.data[lane] + 2U * static_cast<std::uint32_t>(sample) - 2U);
    };
    const __m256 s0 = load(0U, _mm_cvtsi128_si32(index));
    const __m256 s1 = load(1U, _mm_extract_epi32(index, 1));
    const __m256 s2 = load(2U, _mm_extract_epi32(index, 2));
    const __m256 s3 = load(3U, _mm_extract_epi32(index, 3));
    const __m256 low01 = _mm256_unpacklo_ps(s0, s1);  // [s0.r0 s1.r0 s0.i0 s1.i0 | s0.r2 s1.r2 s0.i2 s1.i2]
    const __m256 high01 = _mm256_unpackhi_ps(s0, s1); // [s0.r1 s1.r1 s0.i1 s1.i1 | s0.r3 s1.r3 s0.i3 s1.i3]
    const __m256 low23 = _mm256_unpacklo_ps(s2, s3);
    const __m256 high23 = _mm256_unpackhi_ps(s2, s3);
    const __m256 re02 = _mm256_shuffle_ps(low01, low23, _MM_SHUFFLE(1, 0, 1, 0)); // [re of tap 0 | re of tap 2], lanes 0..3
    const __m256 im02 = _mm256_shuffle_ps(low01, low23, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 re13 = _mm256_shuffle_ps(high01, high23, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 im13 = _mm256_shuffle_ps(high01, high23, _MM_SHUFFLE(3, 2, 3, 2));

    const __m128 one = _mm_set1_ps(1.0f);
    if constexpr (Cubic) {
        // Lagrange weights for the nodes -1, 0, 1, 2, factored so every weight is two products deep.
        const __m128 a = _mm_sub_ps(mu, one);
        const __m128 b = _mm_sub_ps(mu, _mm_set1_ps(2.0f));
        const __m128 c = _mm_add_ps(mu, one);
        const __m128 ab = _mm_mul_ps(a, b);
        const __m128 cm = _mm_mul_ps(c, mu);
        const __m128 c0 = _mm_mul_ps(_mm_mul_ps(mu, _mm_set1_ps(-1.0f / 6.0f)), ab);
        const __m128 c1 = _mm_mul_ps(_mm_mul_ps(c, _mm_set1_ps(0.5f)), ab);
        const __m128 c2 = _mm_mul_ps(cm, _mm_mul_ps(b, _mm_set1_ps(-0.5f)));
        const __m128 c3 = _mm_mul_ps(cm, _mm_mul_ps(a, _mm_set1_ps(1.0f / 6.0f)));
        const __m256 c02 = _mm256_set_m128(c2, c0);
        const __m256 c13 = _mm256_set_m128(c3, c1);
        // (c0 x0 + c1 x1) + (c2 x2 + c3 x3), the two brackets in the two halves.
        const __m256 re = _mm256_fmadd_ps(c13, re13, _mm256_mul_ps(c02, re02));
        const __m256 im = _mm256_fmadd_ps(c13, im13, _mm256_mul_ps(c02, im02));
        return {_mm_add_ps(_mm256_castps256_ps128(re), _mm256_extractf128_ps(re, 1)), _mm_add_ps(_mm256_castps256_ps128(im), _mm256_extractf128_ps(im, 1))};
    } else {
        const __m128 c1 = _mm_sub_ps(one, mu);
        return {_mm_fmadd_ps(mu, _mm256_extractf128_ps(re02, 1), _mm_mul_ps(c1, _mm256_castps256_ps128(re13))),
                _mm_fmadd_ps(mu, _mm256_extractf128_ps(im02, 1), _mm_mul_ps(c1, _mm256_castps256_ps128(im13)))};
    }
}

struct LoopConstants final {
    __m256d limit;
    /** Lowest time whose Gardner mid-point stencil still starts inside the data. */
    __m256d lower_limit;
    __m256d omega_min;
    __m256d omega_max;
    __m256d alpha;
    __m256d beta;
    __m256d alpha_beta;
    /** alpha + beta / 2, see Advance(). */
    __m256d alpha_beta_mid;
};

struct Advanced final {
    __m256d time;
    __m256d omega;
    __m256d mid;
};

/**
 * Loop filter. omega integrates the error, and the next symbol lies omega plus the proportional
 * correction ahead:
 *   omega' = clamp(omega + beta e),  t' = t + omega' + alpha e,  mid' = t' - omega' / 2.
 * Unless the clamp is active these are t' = (t + omega) + (alpha + beta) e and
 * mid' = (t + omega / 2) + (alpha + beta / 2) e: one multiply-add after the error each.
 */
[[nodiscard]] inline Advanced Advance(const LoopConstants& k, const __m256d time, const __m256d omega, const __m256d error) noexcept {
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d omega_raw = _mm256_fmadd_pd(k.beta, error, omega);
    const __m256d omega_clamped = _mm256_min_pd(_mm256_max_pd(omega_raw, k.omega_min), k.omega_max);
    Advanced next{_mm256_fmadd_pd(k.alpha_beta, error, _mm256_add_pd(time, omega)), omega_clamped,
                  _mm256_fmadd_pd(k.alpha_beta_mid, error, _mm256_fmadd_pd(half, omega, time))};
    if (_mm256_movemask_pd(_mm256_cmp_pd(omega_clamped, omega_raw, _CMP_NEQ_UQ)) != 0) [[unlikely]] {
        next.time = _mm256_add_pd(time, _mm256_fmadd_pd(k.alpha, error, omega_clamped));
        next.mid = _mm256_fnmadd_pd(half, omega_clamped, next.time);
    }
    return next;
}

/** Loop-carried values of a batch. */
struct Trajectory final {
    __m256d time;
    __m256d omega;
    Complex4 previous;
};

/**
 * Up to `steps` symbols for all four lanes with one detector, while every lane stays inside
 * [lower_limit, limit); returns the number of steps taken. With `LogErrors` the detector errors
 * and energies are logged for the lock tracking replay.
 *
 * A separate function on purpose: everything the loop carries arrives by value and lives in
 * locals, so it stays in registers. As part of Run(), whose general path indexes the pointer
 * and count arrays per lane, the compiler kept the trajectory and the pointers in memory and
 * put a store-to-load round trip on the recurrence.
 */
template <bool Cubic, bool MuellerMuller, bool LogErrors>
[[gnu::noinline]] std::size_t Batch(const Inputs inputs, const LoopConstants& constants, Trajectory& trajectory, float* output0, float* output1,
                                    float* output2, float* output3, const std::size_t steps, float (*const error_log)[kSymbolSync4Lanes],
                                    float (*const energy_log)[kSymbolSync4Lanes]) noexcept {
    const LoopConstants k = constants;
    __m256d time = trajectory.time;
    __m256d omega = trajectory.omega;
    __m256d mid_time = _mm256_fnmadd_pd(_mm256_set1_pd(0.5), omega, time);
    Complex4 previous = trajectory.previous;
    std::size_t done = 0U;
    do {
        const Complex4 current = Interpolate<Cubic, false>(inputs, time);
        __m128 error4;
        if constexpr (MuellerMuller) {
            error4 = MuellerMullerError(previous, current);
            if constexpr (LogErrors) {
                _mm_store_ps(energy_log[done], DetectorEnergy(previous, current, current, true));
            }
        } else {
            const Complex4 mid = Interpolate<Cubic, false>(inputs, mid_time);
            error4 = GardnerError(previous, mid, current);
            if constexpr (LogErrors) {
                _mm_store_ps(energy_log[done], DetectorEnergy(previous, mid, current, false));
            }
        }
        if constexpr (LogErrors) {
            _mm_store_ps(error_log[done], error4);
        }
        const Advanced next = Advance(k, time, omega, _mm256_cvtps_pd(error4));
        time = next.time;
        omega = next.omega;
        mid_time = next.mid;
        previous = current;
        const __m128 low = _mm_unpacklo_ps(current.re, current.im);
        const __m128 high = _mm_unpackhi_ps(current.re, current.im);
        _mm_storel_pi(reinterpret_cast<__m64*>(output0), low);
        _mm_storeh_pi(reinterpret_cast<__m64*>(output1), low);
        _mm_storel_pi(reinterpret_cast<__m64*>(output2), high);
        _mm_storeh_pi(reinterpret_cast<__m64*>(output3), high);
        output0 += 2;
        output1 += 2;
        output2 += 2;
        output3 += 2;
        ++done;
    } while (done < steps &&
             _mm256_movemask_pd(_mm256_and_pd(_mm256_cmp_pd(time, k.limit, _CMP_LT_OQ), _mm256_cmp_pd(time, k.lower_limit, _CMP_GE_OQ))) == 0xF);
    trajectory = {time, omega, previous};
    return done;
}

template <bool Cubic, bool AutoTed>
void Run(const SymbolSync4Params& params, SymbolSync4State& state, SymbolSync4Segments& segments) noexcept {
    Inputs inputs{};
    alignas(16) std::int32_t last_index[kSymbolSync4Lanes];
    alignas(32) double lower[kSymbolSync4Lanes];
    alignas(32) double mid_gain[kSymbolSync4Lanes];
    for (std::size_t lane = 0U; lane < kSymbolSync4Lanes; ++lane) {
        inputs.data[lane] = segments.data[lane];
        last_index[lane] = static_cast<std::int32_t>(segments.length[lane]) - 3;
        // Lowest time whose Gardner mid-point stencil still starts inside the data.
        lower[lane] = 0.5 * params.omega_max[lane] + 2.0;
        mid_gain[lane] = params.alpha_beta[lane] - 0.5 * params.beta[lane];
    }
    inputs.last_index = _mm_load_si128(reinterpret_cast<const __m128i*>(last_index));

    const LoopConstants constants{.limit = _mm256_loadu_pd(segments.limit),
                                  .lower_limit = _mm256_load_pd(lower),
                                  .omega_min = _mm256_load_pd(params.omega_min),
                                  .omega_max = _mm256_load_pd(params.omega_max),
                                  .alpha = _mm256_load_pd(params.alpha),
                                  .beta = _mm256_load_pd(params.beta),
                                  .alpha_beta = _mm256_load_pd(params.alpha_beta),
                                  .alpha_beta_mid = _mm256_load_pd(mid_gain)};
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d one = _mm256_set1_pd(1.0);

    __m256d time = _mm256_load_pd(state.time);
    __m256d omega = _mm256_load_pd(state.omega);
    Complex4 previous = Deinterleave(state.previous);
    __m256d symbols = _mm256_load_pd(state.symbols);
    __m256d ema = _mm256_load_pd(state.err_norm2_ema);
    __m256d lock_run = _mm256_load_pd(state.lock_run);
    __m256d unlock_run = _mm256_load_pd(state.unlock_run);
    int have_previous = LaneBits(state.have_previous);
    int mueller_muller_bits = LaneBits(state.mueller_muller);

    float* output[kSymbolSync4Lanes];
    std::size_t count[kSymbolSync4Lanes];
    for (std::size_t lane = 0U; lane < kSymbolSync4Lanes; ++lane) {
        output[lane] = segments.output[lane];
        count[lane] = segments.output_count[lane];
    }

    const __m256d ema_alpha = _mm256_set1_pd(params.lock_ema_alpha);
    const __m256d one_minus_ema_alpha = _mm256_set1_pd(1.0 - params.lock_ema_alpha);
    const __m256d mm_min_symbols = _mm256_set1_pd(params.mm_min_symbols);
    const __m256d mm_hold = _mm256_set1_pd(params.mm_hold_symbols);
    const __m256d fallback_hold = _mm256_set1_pd(params.fallback_hold_symbols);
    const __m256d mm_threshold = _mm256_set1_pd(params.mm_threshold);
    const __m256d fallback_threshold = _mm256_set1_pd(params.fallback_threshold);
    const __m256d fallback_enabled = _mm256_castsi256_pd(_mm256_set1_epi64x(params.auto_ted_fallback ? -1 : 0));

    // Lock tracking of one step for the lanes in `update`; reports the lanes that switch detector.
    const auto track_lock = [&](const __m128 error4, const __m128 energy4, const __m256d symbols_before, const __m256d mueller_muller,
                                const __m256d update) noexcept {
        const __m128 normalised = _mm_div_ps(error4, _mm_add_ps(energy4, _mm_set1_ps(1e-12f)));
        const __m256d norm2 = _mm256_cvtps_pd(_mm_mul_ps(normalised, normalised));
        // (1 - a) * ema + a * norm2 with a single multiply-add on the ema recurrence.
        ema = _mm256_blendv_pd(ema, _mm256_fmadd_pd(one_minus_ema_alpha, ema, _mm256_mul_pd(ema_alpha, norm2)), update);

        const __m256d gardner_lanes = _mm256_and_pd(_mm256_andnot_pd(mueller_muller, update), _mm256_cmp_pd(symbols_before, mm_min_symbols, _CMP_GE_OQ));
        const __m256d locked = _mm256_and_pd(_mm256_add_pd(lock_run, one), _mm256_cmp_pd(ema, mm_threshold, _CMP_LT_OQ));
        lock_run = _mm256_blendv_pd(lock_run, locked, gardner_lanes);
        const __m256d to_mm = _mm256_and_pd(gardner_lanes, _mm256_cmp_pd(lock_run, mm_hold, _CMP_GE_OQ));

        const __m256d mm_lanes = _mm256_and_pd(_mm256_and_pd(mueller_muller, update), fallback_enabled);
        const __m256d unlocked = _mm256_and_pd(_mm256_add_pd(unlock_run, one), _mm256_cmp_pd(ema, fallback_threshold, _CMP_GT_OQ));
        unlock_run = _mm256_blendv_pd(unlock_run, unlocked, mm_lanes);
        const __m256d to_gardner = _mm256_and_pd(mm_lanes, _mm256_cmp_pd(unlock_run, fallback_hold, _CMP_GE_OQ));
        return _mm256_movemask_pd(_mm256_or_pd(to_mm, to_gardner));
    };

    const auto switch_detectors = [&](const int switching) noexcept {
        _mm256_store_pd(state.err_norm2_ema, ema);
        _mm256_store_pd(state.lock_run, lock_run);
        _mm256_store_pd(state.unlock_run, unlock_run);
        for (std::size_t lane = 0U; lane < kSymbolSync4Lanes; ++lane) {
            state.mueller_muller[lane] = (mueller_muller_bits >> lane) & 1;
            if ((switching >> lane) & 1) {
                SymbolSync4SwitchTed(state, lane);
            }
        }
        ema = _mm256_load_pd(state.err_norm2_ema);
        lock_run = _mm256_load_pd(state.lock_run);
        unlock_run = _mm256_load_pd(state.unlock_run);
        mueller_muller_bits = LaneBits(state.mueller_muller);
    };

    // General step: one symbol for every lane in `active`, any mix of detectors, first symbols,
    // clamped stencils and per-step lock tracking. Used at segment edges and around detector
    // switches only.
    const auto masked_step = [&](const int active) noexcept {
        const Complex4 current = Interpolate<Cubic, true>(inputs, time);
        const Complex4 mid = Interpolate<Cubic, true>(inputs, _mm256_fnmadd_pd(half, omega, time));
        const __m128 mm4 = LaneMask4(mueller_muller_bits);
        const __m128 error4 = _mm_and_ps(_mm_blendv_ps(GardnerError(previous, mid, current), MuellerMullerError(previous, current), mm4),
                                         LaneMask4(have_previous)); // the first symbol has no error
        const Advanced next = Advance(constants, time, omega, _mm256_cvtps_pd(error4));

        if constexpr (AutoTed) {
            const __m128 energy4 = _mm_blendv_ps(DetectorEnergy(previous, mid, current, false), DetectorEnergy(previous, mid, current, true), mm4);
            const int switching = track_lock(error4, energy4, symbols, LaneMask(mueller_muller_bits), LaneMask(active & have_previous));
            if (switching != 0) [[unlikely]] {
                switch_detectors(switching);
            }
        }

        const __m256d active_mask = LaneMask(active);
        time = _mm256_blendv_pd(time, next.time, active_mask);
        omega = _mm256_blendv_pd(omega, next.omega, active_mask);
        previous = Select(previous, current, LaneMask4(active));
        symbols = _mm256_add_pd(symbols, _mm256_and_pd(active_mask, one));
        have_previous |= active;
        alignas(16) float pairs[2U * kSymbolSync4Lanes];
        Interleave(current, pairs);
        for (std::size_t lane = 0U; lane < kSymbolSync4Lanes; ++lane) {
            if ((active >> lane) & 1) {
                output[lane][2U * count[lane]] = pairs[2U * lane];
                output[lane][2U * count[lane] + 1U] = pairs[2U * lane + 1U];
                ++count[lane];
            }
        }
    };

    // Lock tracking of a batch, replayed from the logged errors. No lane can reach its hold count
    // within a batch (see the batch sizing below), so this gives exactly the per-step result
    // while only the few operations of the smoothing recurrence stay serial.
    alignas(32) float error_log[kBatch][kSymbolSync4Lanes];
    alignas(32) float energy_log[kBatch][kSymbolSync4Lanes];
    const auto replay_lock = [&]<bool MuellerMuller>(const std::size_t steps) noexcept {
        // The squared normalised errors do not depend on each other: compute them two steps per
        // vector up front (in place of the errors), leaving only the recurrences serial.
        for (std::size_t index = 0U; index < steps; index += 2U) {
            const __m256 normalised = _mm256_div_ps(_mm256_load_ps(error_log[index]), _mm256_add_ps(_mm256_load_ps(energy_log[index]), _mm256_set1_ps(1e-12f)));
            _mm256_store_ps(error_log[index], _mm256_mul_ps(normalised, normalised));
        }
        const auto norm2 = [&](const std::size_t index) noexcept { return _mm256_mul_pd(ema_alpha, _mm256_cvtps_pd(_mm_load_ps(error_log[index]))); };
        __m256d smoothed = ema;
        if constexpr (MuellerMuller) {
            __m256d run = unlock_run;
            for (std::size_t index = 0U; index < steps; ++index) {
                smoothed = _mm256_fmadd_pd(one_minus_ema_alpha, smoothed, norm2(index));
                run = _mm256_and_pd(_mm256_add_pd(run, one), _mm256_cmp_pd(smoothed, fallback_threshold, _CMP_GT_OQ));
            }
            if (params.auto_ted_fallback) {
                unlock_run = run;
            }
        } else {
            __m256d run = lock_run;
            __m256d symbols_before = symbols;
            for (std::size_t index = 0U; index < steps; ++index) {
                smoothed = _mm256_fmadd_pd(one_minus_ema_alpha, smoothed, norm2(index));
                const __m256d eligible = _mm256_cmp_pd(symbols_before, mm_min_symbols, _CMP_GE_OQ);
                run = _mm256_blendv_pd(run, _mm256_and_pd(_mm256_add_pd(run, one), _mm256_cmp_pd(smoothed, mm_threshold, _CMP_LT_OQ)), eligible);
                symbols_before = _mm256_add_pd(symbols_before, one);
            }
            lock_run = run;
        }
        ema = smoothed;
    };

    const auto batch = [&]<bool MuellerMuller>(const std::size_t steps) noexcept {
        Trajectory trajectory{time, omega, previous};
        const std::size_t done = Batch<Cubic, MuellerMuller, AutoTed>(inputs, constants, trajectory, output[0] + 2U * count[0], output[1] + 2U * count[1],
                                                                      output[2] + 2U * count[2], output[3] + 2U * count[3], steps, error_log, energy_log);
        time = trajectory.time;
        omega = trajectory.omega;
        previous = trajectory.previous;
        for (std::size_t lane = 0U; lane < kSymbolSync4Lanes; ++lane) {
            count[lane] += done;
        }
        if constexpr (AutoTed) {
            replay_lock.template operator()<MuellerMuller>(done);
        }
        symbols = _mm256_add_pd(symbols, _mm256_set1_pd(static_cast<double>(done)));
    };

    for (;;) {
        int room = 0;
        std::size_t budget = kBatch;
        for (std::size_t lane = 0U; lane < kSymbolSync4Lanes; ++lane) {
            const std::size_t left = segments.output_capacity[lane] - count[lane];
            room |= left != 0U ? 1 << lane : 0;
            budget = std::min(budget, left);
        }
        const int in_range = _mm256_movemask_pd(_mm256_cmp_pd(time, constants.limit, _CMP_LT_OQ));
        const int active = in_range & room;
        if (active == 0) {
            break;
        }
        const bool uniform = mueller_muller_bits == 0 || mueller_muller_bits == 0xF;
        if (active == 0xF && have_previous == 0xF && uniform && _mm256_movemask_pd(_mm256_cmp_pd(time, constants.lower_limit, _CMP_GE_OQ)) == 0xF) {
            if constexpr (AutoTed) {
                // A lane switches once its run count reaches the hold count, and a run grows by
                // at most one per symbol: keep batches short enough that no lane gets there.
                const bool mm = mueller_muller_bits != 0;
                if (!mm || params.auto_ted_fallback) {
                    const double hold = mm ? params.fallback_hold_symbols : params.mm_hold_symbols;
                    const double run = HorizontalMax(mm ? unlock_run : lock_run);
                    budget = hold - run > static_cast<double>(budget) ? budget : static_cast<std::size_t>(std::max(0.0, hold - run - 1.0));
                }
            }
            if (budget != 0U) {
                mueller_muller_bits != 0 ? batch.template operator()<true>(budget) : batch.template operator()<false>(budget);
                continue;
            }
        }
        masked_step(active);
    }

    _mm256_store_pd(state.time, time);
    _mm256_store_pd(state.omega, omega);
    Interleave(previous, state.previous);
    _mm256_store_pd(state.symbols, symbols);
    _mm256_store_pd(state.err_norm2_ema, ema);
    _mm256_store_pd(state.lock_run, lock_run);
    _mm256_store_pd(state.unlock_run, unlock_run);
    for (std::size_t lane = 0U; lane < kSymbolSync4Lanes; ++lane) {
        state.have_previous[lane] = (have_previous >> lane) & 1;
        state.mueller_muller[lane] = (mueller_muller_bits >> lane) & 1;
        segments.output_count[lane] = count[lane];
    }
}

} // namespace

void SymbolSync4_avx2(const SymbolSync4Params& params, SymbolSync4State& state, SymbolSync4Segments& segments) noexcept {
    if (params.cubic) {
        params.auto_ted ? Run<true, true>(params, state, segments) : Run<true, false>(params, state, segments);
    } else {
        params.auto_ted ? Run<false, true>(params, state, segments) : Run<false, false>(params, state, segments);
    }
}

} // namespace uni::simd::kernels

#endif
