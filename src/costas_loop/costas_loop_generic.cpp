#include "costas_loop_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace uni::simd::kernels {

namespace {
[[nodiscard]] constexpr float Infinity() noexcept { return std::numeric_limits<float>::infinity(); }
} // namespace

void QpskCostas4_generic(uni_simd_qpsk_costas4_t& kernel,
                         const uni_simd_qpsk_costas4_block_t& block) noexcept {
    for (std::size_t sample = 0U; sample < block.sample_count; ++sample) {
        for (std::size_t lane = 0U; lane < UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT; ++lane) {
            auto* const data = block.channels[lane] + sample * 2U;
            const float real = data[0] * block.input_gain[lane];
            const float imag = data[1] * block.input_gain[lane];
            const float phase_cos = kernel.state.phase_cos[lane];
            const float phase_sin = kernel.state.phase_sin[lane];
            const float output_real = std::fma(real, phase_cos, imag * phase_sin);
            const float output_imag = std::fma(imag, phase_cos, -(real * phase_sin));
            data[0] = output_real;
            data[1] = output_imag;

            float error = std::copysign(1.0f, output_real) * output_imag -
                          std::copysign(1.0f, output_imag) * output_real;
            // A disabled bound becomes infinite so the clamp stays branch-free and exact.
            const float clip_bound = kernel.config.error_clip[lane] > 0.0f ? kernel.config.error_clip[lane] : Infinity();
            error = std::min(std::max(error, -clip_bound), clip_bound);
            kernel.state.last_error[lane] = error;

            const float limit_bound = block.frequency_limit[lane] > 0.0f ? block.frequency_limit[lane] : Infinity();
            const float frequency =
                std::min(std::max(std::fma(kernel.config.beta[lane], error, kernel.state.frequency[lane]), -limit_bound), limit_bound);
            kernel.state.frequency[lane] = frequency;
            const float delta = std::fma(kernel.config.alpha[lane], error, frequency);

            // Advance the phasor by the oldest pending step, then queue this sample's step.
            auto& state = kernel.state;
            constexpr std::size_t newest = UNI_SIMD_QPSK_COSTAS4_LOOP_DELAY - 1U;
            const float step_cos = state.pending_step_cos[0][lane];
            const float step_sin = state.pending_step_sin[0][lane];
            state.phase_cos[lane] = std::fma(phase_cos, step_cos, -(phase_sin * step_sin));
            state.phase_sin[lane] = std::fma(phase_sin, step_cos, phase_cos * step_sin);
            state.phase[lane] += state.pending_step[0][lane];
            for (std::size_t delay = 0U; delay < newest; ++delay) {
                state.pending_step[delay][lane] = state.pending_step[delay + 1U][lane];
                state.pending_step_cos[delay][lane] = state.pending_step_cos[delay + 1U][lane];
                state.pending_step_sin[delay][lane] = state.pending_step_sin[delay + 1U][lane];
            }
            state.pending_step[newest][lane] = delta;
            Costas4SinCos(delta, state.pending_step_sin[newest][lane], state.pending_step_cos[newest][lane]);
        }
        if (++kernel.samples_since_normalization == 512U) {
            kernel.samples_since_normalization = 0U;
            for (std::size_t lane = 0U; lane < UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT; ++lane) {
                Costas4Normalize(kernel.state.phase[lane], kernel.state.phase_cos[lane], kernel.state.phase_sin[lane]);
            }
        }
    }
}

} // namespace uni::simd::kernels
