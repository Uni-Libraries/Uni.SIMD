#pragma once

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <uni/simd/pfb_channelizer.hpp>

namespace uni::simd::detail {

struct PfbChannelizerAccess final {
    [[nodiscard]] static std::uint64_t signature(const PfbChannelizerPlan& plan) noexcept { return plan.signature_; }
    [[nodiscard]] static std::size_t rows(const PfbChannelizerPlan& plan) noexcept { return plan.row_count_; }
    [[nodiscard]] static std::size_t history_size(const PfbChannelizerPlan& plan) noexcept { return plan.history_size_; }
    [[nodiscard]] static std::size_t phase_period(const PfbChannelizerPlan& plan) noexcept { return plan.phase_period_; }
    [[nodiscard]] static const float* coefficients(const PfbChannelizerPlan& plan) noexcept { return plan.coefficients_.data(); }
    [[nodiscard]] static const float* reversed_coefficients(const PfbChannelizerPlan& plan) noexcept {
        return plan.reversed_coefficients_.data();
    }
    [[nodiscard]] static const float* branch_rotation_re(const PfbChannelizerPlan& plan) noexcept {
        return plan.branch_rotation_re_.data();
    }
    [[nodiscard]] static const float* branch_rotation_im(const PfbChannelizerPlan& plan) noexcept {
        return plan.branch_rotation_im_.data();
    }
    [[nodiscard]] static const float* fft_twiddle_re(const PfbChannelizerPlan& plan) noexcept {
        return plan.fft_twiddle_re_.data();
    }
    [[nodiscard]] static const float* fft_twiddle_im(const PfbChannelizerPlan& plan) noexcept {
        return plan.fft_twiddle_im_.data();
    }
    [[nodiscard]] static const std::uint8_t* bit_reverse(const PfbChannelizerPlan& plan) noexcept {
        return plan.bit_reverse_.data();
    }
    [[nodiscard]] static const float* post_phase_re(const PfbChannelizerPlan& plan) noexcept {
        return plan.post_phase_re_.data();
    }
    [[nodiscard]] static const float* post_phase_im(const PfbChannelizerPlan& plan) noexcept {
        return plan.post_phase_im_.data();
    }

    [[nodiscard]] static std::uint64_t state_signature(const PfbChannelizerState& state) noexcept {
        return state.plan_signature_;
    }
    [[nodiscard]] static std::size_t cursor(const PfbChannelizerState& state) noexcept { return state.cursor_; }
    [[nodiscard]] static std::size_t decimation_phase(const PfbChannelizerState& state) noexcept {
        return state.decimation_phase_;
    }
    [[nodiscard]] static std::size_t post_phase(const PfbChannelizerState& state) noexcept { return state.post_phase_; }
    [[nodiscard]] static const float* history_i(const PfbChannelizerState& state) noexcept { return state.history_i_.data(); }
    [[nodiscard]] static const float* history_q(const PfbChannelizerState& state) noexcept { return state.history_q_.data(); }
    [[nodiscard]] static float* history_i(PfbChannelizerState& state) noexcept { return state.history_i_.data(); }
    [[nodiscard]] static float* history_q(PfbChannelizerState& state) noexcept { return state.history_q_.data(); }

    static void set_cursor(PfbChannelizerState& state, const std::size_t value) noexcept {
        state.cursor_ = static_cast<std::uint16_t>(value);
    }
    static void set_decimation_phase(PfbChannelizerState& state, const std::size_t value) noexcept {
        state.decimation_phase_ = static_cast<std::uint16_t>(value);
    }
    static void set_post_phase(PfbChannelizerState& state, const std::size_t value) noexcept {
        state.post_phase_ = static_cast<std::uint8_t>(value);
    }
};

using PfbChannelizerFn = std::size_t (*)(const PfbChannelizerPlan&, PfbChannelizerState&,
                                         const PfbChannelizerBlockView&) noexcept;

[[nodiscard]] std::size_t PfbChannelizer_generic(const PfbChannelizerPlan& plan, PfbChannelizerState& state,
                                                 const PfbChannelizerBlockView& block) noexcept;
[[nodiscard]] std::size_t PfbChannelizer_avx2fma(const PfbChannelizerPlan& plan, PfbChannelizerState& state,
                                                 const PfbChannelizerBlockView& block) noexcept;
[[nodiscard]] std::size_t PfbChannelizer_neon(const PfbChannelizerPlan& plan, PfbChannelizerState& state,
                                              const PfbChannelizerBlockView& block) noexcept;

[[nodiscard]] inline bool pfb_plan_is_valid(const PfbChannelizerPlan& plan) noexcept {
    const std::size_t bins = plan.bin_count();
    const std::size_t decimation = plan.decimation();
    const std::size_t taps = plan.tap_count();
    const std::size_t rows = PfbChannelizerAccess::rows(plan);
    const std::size_t history = PfbChannelizerAccess::history_size(plan);
    const std::size_t period = PfbChannelizerAccess::phase_period(plan);
    return plan.initialized() && (bins == 4U || bins == 8U || bins == 16U || bins == 32U) &&
           decimation != 0U && bins % decimation == 0U && taps != 0U && taps <= pfb_channelizer_max_taps &&
           rows == (taps + bins - 1U) / bins && history >= rows * bins && history <= pfb_channelizer_max_history &&
           (history & (history - 1U)) == 0U && plan.selected_output_count() <= std::min(bins, pfb_channelizer_max_outputs) &&
           period != 0U && period <= pfb_channelizer_max_phase_period;
}

[[nodiscard]] inline bool pfb_state_is_valid(const PfbChannelizerPlan& plan,
                                             const PfbChannelizerState& state) noexcept {
    return state.initialized() && PfbChannelizerAccess::state_signature(state) == PfbChannelizerAccess::signature(plan) &&
           PfbChannelizerAccess::cursor(state) < PfbChannelizerAccess::history_size(plan) &&
           PfbChannelizerAccess::decimation_phase(state) < plan.decimation() &&
           PfbChannelizerAccess::post_phase(state) < PfbChannelizerAccess::phase_period(plan);
}

[[nodiscard]] inline std::size_t pfb_output_count_unchecked(const PfbChannelizerPlan& plan,
                                                            const PfbChannelizerState& state,
                                                            const std::size_t input_count) noexcept {
    if (input_count == 0U) {
        return 0U;
    }
    const std::size_t phase = PfbChannelizerAccess::decimation_phase(state);
    const std::size_t first = phase == 0U ? 0U : plan.decimation() - phase;
    return first >= input_count ? 0U : 1U + (input_count - 1U - first) / plan.decimation();
}

inline void pfb_inverse_fft_in_place(const PfbChannelizerPlan& plan, float* real, float* imag) noexcept {
    const std::size_t bins = plan.bin_count();
    const auto* bit_reverse = PfbChannelizerAccess::bit_reverse(plan);
    for (std::size_t index = 0U; index < bins; ++index) {
        const std::size_t reversed = bit_reverse[index];
        if (index < reversed) {
            std::swap(real[index], real[reversed]);
            std::swap(imag[index], imag[reversed]);
        }
    }

    const auto* twiddle_re = PfbChannelizerAccess::fft_twiddle_re(plan);
    const auto* twiddle_im = PfbChannelizerAccess::fft_twiddle_im(plan);
    for (std::size_t length = 2U; length <= bins; length *= 2U) {
        const std::size_t half = length / 2U;
        const std::size_t twiddle_step = bins / length;
        for (std::size_t base = 0U; base < bins; base += length) {
            for (std::size_t index = 0U; index < half; ++index) {
                const std::size_t twiddle = index * twiddle_step;
                const std::size_t even = base + index;
                const std::size_t odd = even + half;
                const float odd_re = real[odd];
                const float odd_im = imag[odd];
                const float product_re = odd_re * twiddle_re[twiddle] - odd_im * twiddle_im[twiddle];
                const float product_im = odd_re * twiddle_im[twiddle] + odd_im * twiddle_re[twiddle];
                const float even_re = real[even];
                const float even_im = imag[even];
                real[even] = even_re + product_re;
                imag[even] = even_im + product_im;
                real[odd] = even_re - product_re;
                imag[odd] = even_im - product_im;
            }
        }
    }
}

[[nodiscard]] inline std::complex<float> pfb_apply_post_phase(const PfbChannelizerPlan& plan,
                                                              const std::size_t output,
                                                              const std::size_t phase,
                                                              const float real,
                                                              const float imag) noexcept {
    const std::size_t table_index = output * pfb_channelizer_max_phase_period + phase;
    const float phase_re = PfbChannelizerAccess::post_phase_re(plan)[table_index];
    const float phase_im = PfbChannelizerAccess::post_phase_im(plan)[table_index];

    // The M=8,D=4 phase table contains only exact quarter turns. Keep those corrections to sign/swap operations.
    if (phase_im == 0.0f && (phase_re == 1.0f || phase_re == -1.0f)) {
        return {phase_re * real, phase_re * imag};
    }
    if (phase_re == 0.0f && (phase_im == 1.0f || phase_im == -1.0f)) {
        return {-phase_im * imag, phase_im * real};
    }
    return {real * phase_re - imag * phase_im, real * phase_im + imag * phase_re};
}

} // namespace uni::simd::detail
