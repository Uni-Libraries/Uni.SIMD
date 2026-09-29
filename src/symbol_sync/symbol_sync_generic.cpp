#include "symbol_sync_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace uni::simd::kernels {

namespace {

struct Complex final {
    float re;
    float im;
};

/** cvttpd2dq semantics: truncation toward zero, INT32_MIN for NaN and out-of-range values. */
[[nodiscard]] std::int32_t TruncateToInt32(const double value) noexcept {
    return value > -2147483649.0 && value < 2147483648.0 ? static_cast<std::int32_t>(value) : std::numeric_limits<std::int32_t>::min();
}

/**
 * Interpolates at `position` from the four samples around it. The operation order matches the
 * vector backend: index by truncation, mu from floor(), weights in float, two products per half.
 */
[[nodiscard]] Complex Interpolate(const float* const data, const std::size_t length, const double position, const bool cubic) noexcept {
    const std::int32_t index = std::min(std::max(TruncateToInt32(position), 1), static_cast<std::int32_t>(length) - 3);
    const auto mu = static_cast<float>(position - std::floor(position));
    const float* const x = data + 2 * static_cast<std::ptrdiff_t>(index - 1);
    if (cubic) {
        const float a = mu - 1.0f;
        const float b = mu - 2.0f;
        const float c = mu + 1.0f;
        const float ab = a * b;
        const float cm = c * mu;
        const float c0 = (mu * (-1.0f / 6.0f)) * ab;
        const float c1 = (c * 0.5f) * ab;
        const float c2 = cm * (b * -0.5f);
        const float c3 = cm * (a * (1.0f / 6.0f));
        return {std::fma(c1, x[2], c0 * x[0]) + std::fma(c3, x[6], c2 * x[4]), std::fma(c1, x[3], c0 * x[1]) + std::fma(c3, x[7], c2 * x[5])};
    }
    const float c1 = 1.0f - mu;
    return {std::fma(mu, x[4], c1 * x[2]), std::fma(mu, x[5], c1 * x[3])};
}

/** QPSK slicer times value: sign(decision) * value, with -0 counted as negative like the vector backend's sign-bit flip. */
[[nodiscard]] float SignTimes(const float decision, const float value) noexcept { return std::signbit(decision) ? -value : value; }

/** maxpd / minpd semantics (the second operand wins when either is NaN). */
[[nodiscard]] double Max(const double value, const double bound) noexcept { return value > bound ? value : bound; }
[[nodiscard]] double Min(const double value, const double bound) noexcept { return value < bound ? value : bound; }

} // namespace

void SymbolSync4_generic(const SymbolSync4Params& params, SymbolSync4State& state, SymbolSync4Segments& segments) noexcept {
    const double one_minus_ema_alpha = 1.0 - params.lock_ema_alpha;
    // Lanes are independent, so the scalar reference simply runs them one after another.
    for (std::size_t lane = 0U; lane < kSymbolSync4Lanes; ++lane) {
        const float* const data = segments.data[lane];
        const std::size_t length = segments.length[lane];
        const double limit = segments.limit[lane];
        float* const output = segments.output[lane];
        const std::size_t capacity = segments.output_capacity[lane];
        std::size_t count = segments.output_count[lane];

        double time = state.time[lane];
        double omega = state.omega[lane];
        Complex previous{state.previous[2U * lane], state.previous[2U * lane + 1U]};
        bool have_previous = state.have_previous[lane] != 0U;

        while (count < capacity && time < limit) {
            const Complex current = Interpolate(data, length, time, params.cubic);
            const Complex mid = Interpolate(data, length, time - 0.5 * omega, params.cubic);
            const bool mueller_muller = state.mueller_muller[lane] != 0U;

            float error = 0.0f;
            if (have_previous) {
                error = mueller_muller ? (SignTimes(previous.re, current.re) - SignTimes(current.re, previous.re)) +
                                             (SignTimes(previous.im, current.im) - SignTimes(current.im, previous.im))
                                       : mid.re * (previous.re - current.re) + mid.im * (previous.im - current.im);
            }
            const auto error_d = static_cast<double>(error);
            const double omega_raw = std::fma(params.beta[lane], error_d, omega);
            const double omega_next = Min(Max(omega_raw, params.omega_min[lane]), params.omega_max[lane]);
            double time_next = std::fma(params.alpha_beta[lane], error_d, time + omega);
            if (omega_next != omega_raw) {
                time_next = time + std::fma(params.alpha[lane], error_d, omega_next);
            }

            if (params.auto_ted && have_previous) {
                float power_re = std::fma(previous.re, previous.re, current.re * current.re);
                float power_im = std::fma(previous.im, previous.im, current.im * current.im);
                if (!mueller_muller) {
                    power_re = std::fma(mid.re, mid.re, power_re);
                    power_im = std::fma(mid.im, mid.im, power_im);
                }
                const float normalised = error / ((power_re + power_im) + 1e-12f);
                const auto norm2 = static_cast<double>(normalised * normalised);
                double& ema = state.err_norm2_ema[lane];
                ema = std::fma(one_minus_ema_alpha, ema, params.lock_ema_alpha * norm2);
                if (!mueller_muller) {
                    if (state.symbols[lane] >= params.mm_min_symbols) {
                        state.lock_run[lane] = ema < params.mm_threshold ? state.lock_run[lane] + 1.0 : 0.0;
                        if (state.lock_run[lane] >= params.mm_hold_symbols) {
                            SymbolSync4SwitchTed(state, lane);
                        }
                    }
                } else if (params.auto_ted_fallback) {
                    state.unlock_run[lane] = ema > params.fallback_threshold ? state.unlock_run[lane] + 1.0 : 0.0;
                    if (state.unlock_run[lane] >= params.fallback_hold_symbols) {
                        SymbolSync4SwitchTed(state, lane);
                    }
                }
            }

            time = time_next;
            omega = omega_next;
            previous = current;
            have_previous = true;
            output[2U * count] = current.re;
            output[2U * count + 1U] = current.im;
            ++count;
            state.symbols[lane] += 1.0;
        }

        state.time[lane] = time;
        state.omega[lane] = omega;
        state.previous[2U * lane] = previous.re;
        state.previous[2U * lane + 1U] = previous.im;
        state.have_previous[lane] = have_previous ? 1U : 0U;
        segments.output_count[lane] = count;
    }
}

} // namespace uni::simd::kernels
