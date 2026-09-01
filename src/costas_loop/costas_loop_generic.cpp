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
            float delta_sin = 0.0f;
            float delta_cos = 1.0f;
            Costas4SinCos(delta, delta_sin, delta_cos);
            const float previous_cos = kernel.state.phase_cos[lane];
            const float previous_sin = kernel.state.phase_sin[lane];
            kernel.state.phase_cos[lane] = std::fma(previous_cos, delta_cos, -(previous_sin * delta_sin));
            kernel.state.phase_sin[lane] = std::fma(previous_sin, delta_cos, previous_cos * delta_sin);
            kernel.state.phase[lane] += delta;
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
