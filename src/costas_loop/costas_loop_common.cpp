#include "costas_loop_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>

namespace uni::simd::kernels {

void Costas4Normalize(float& phase, float& phase_cos, float& phase_sin) noexcept {
    constexpr float pi = 3.14159265358979323846f;
    constexpr float two_pi = 2.0f * pi;
    phase = std::remainder(phase, two_pi);
    if (phase <= -pi) {
        phase = pi;
    }
    const float norm2 = phase_cos * phase_cos + phase_sin * phase_sin;
    if (std::isfinite(norm2) && norm2 > 0.0f) {
        const float inverse_norm = 1.0f / std::sqrt(norm2);
        phase_cos *= inverse_norm;
        phase_sin *= inverse_norm;
    } else {
        phase_cos = std::cos(phase);
        phase_sin = std::sin(phase);
    }
}

void Costas4SinCos(const float delta, float& phase_sin, float& phase_cos) noexcept {
    constexpr float approximation_limit = 0.25f;
    if (std::abs(delta) <= approximation_limit) {
        // Horner form kept in sync with the vector kernels: the odd factor is folded into
        // the final fused multiply-add so every backend rounds identically.
        const float squared = delta * delta;
        const float sin_polynomial = std::fma(squared, 1.0f / 120.0f, -1.0f / 6.0f);
        const float cos_polynomial = std::fma(squared, std::fma(squared, 1.0f / 24.0f, -0.5f), 1.0f);
        phase_sin = std::fma(delta * squared, sin_polynomial, delta);
        phase_cos = cos_polynomial;
    } else {
        phase_sin = std::sin(delta);
        phase_cos = std::cos(delta);
    }
}

[[nodiscard]] static bool ValidBackend(const uni_simd_backend_e backend) noexcept {
    return backend <= UNI_SIMD_BACKEND_AARCH64_NEON;
}

[[nodiscard]] bool ValidConfig(const uni_simd_qpsk_costas4_config_t& config) noexcept {
    if (config.descriptor_size != UNI_SIMD_QPSK_COSTAS4_CONFIG_DESCRIPTOR_SIZE) {
        return false;
    }
    for (std::size_t lane = 0U; lane < UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT; ++lane) {
        if (!std::isfinite(config.alpha[lane]) || !std::isfinite(config.beta[lane]) ||
            !std::isfinite(config.error_clip[lane]) || config.error_clip[lane] < 0.0f ||
            !std::isfinite(config.initial_phase[lane]) || !std::isfinite(config.initial_frequency[lane])) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] static bool NaturallyAligned(const void* const pointer) noexcept {
    return reinterpret_cast<std::uintptr_t>(pointer) % alignof(float) == 0U;
}

[[nodiscard]] static bool Overlaps(const void* const left, const std::size_t left_bytes,
                                   const void* const right, const std::size_t right_bytes) noexcept {
    if (left_bytes == 0U || right_bytes == 0U) {
        return false;
    }
    const auto left_begin = reinterpret_cast<std::uintptr_t>(left);
    const auto right_begin = reinterpret_cast<std::uintptr_t>(right);
    return left_begin <= right_begin ? right_begin - left_begin < left_bytes
                                     : left_begin - right_begin < right_bytes;
}

uni_simd_result_e QpskCostas4Initialize(
    uni_simd_qpsk_costas4_t& kernel,
    const uni_simd_qpsk_costas4_config_t& config,
    const uni_simd_backend_e requested_backend,
    const uni_simd_math_mode_e math_mode) noexcept {
    if (!ValidBackend(requested_backend) || math_mode > UNI_SIMD_MATH_DETERMINISTIC ||
        !ValidConfig(config)) {
        return UNI_SIMD_RESULT_INVALID_ARGUMENT;
    }
    kernel = {};
    kernel.config = config;
    kernel.initial_state.descriptor_size = UNI_SIMD_QPSK_COSTAS4_STATE_DESCRIPTOR_SIZE;
    for (std::size_t lane = 0U; lane < UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT; ++lane) {
        kernel.initial_state.phase[lane] = std::remainder(config.initial_phase[lane], 2.0f * 3.14159265358979323846f);
        kernel.initial_state.phase_cos[lane] = std::cos(kernel.initial_state.phase[lane]);
        kernel.initial_state.phase_sin[lane] = std::sin(kernel.initial_state.phase[lane]);
        kernel.initial_state.frequency[lane] = config.initial_frequency[lane];
        for (std::size_t delay = 0U; delay < UNI_SIMD_QPSK_COSTAS4_LOOP_DELAY; ++delay) {
            kernel.initial_state.pending_step[delay][lane] = 0.0f;
            kernel.initial_state.pending_step_cos[delay][lane] = 1.0f;
            kernel.initial_state.pending_step_sin[delay][lane] = 0.0f;
        }
    }
    kernel.state = kernel.initial_state;
    kernel.process = &QpskCostas4_generic;
    kernel.backend = UNI_SIMD_BACKEND_GENERIC;

    const bool deterministic = math_mode == UNI_SIMD_MATH_DETERMINISTIC;
    const auto& capabilities = uni::simd::capabilities();
#if UNI_SIMD_HAVE_AVX2_FMA
    const bool avx2_available = capabilities.avx2 && capabilities.fma;
    const bool avx2_requested = requested_backend == UNI_SIMD_BACKEND_AUTOMATIC ||
                                requested_backend == UNI_SIMD_BACKEND_X86_AVX2_FMA ||
                                requested_backend == UNI_SIMD_BACKEND_X86_AVX512;
    if (!deterministic && avx2_available && avx2_requested) {
        kernel.process = &QpskCostas4_avx2;
        kernel.backend = UNI_SIMD_BACKEND_X86_AVX2_FMA;
    }
#endif
    if (requested_backend != UNI_SIMD_BACKEND_AUTOMATIC && requested_backend != UNI_SIMD_BACKEND_GENERIC &&
        kernel.backend == UNI_SIMD_BACKEND_GENERIC && !deterministic) {
        return UNI_SIMD_RESULT_UNSUPPORTED_BACKEND;
    }
    return UNI_SIMD_RESULT_SUCCESS;
}

uni_simd_result_e QpskCostas4Reset(uni_simd_qpsk_costas4_t& kernel) noexcept {
    kernel.state = kernel.initial_state;
    kernel.samples_since_normalization = 0U;
    return UNI_SIMD_RESULT_SUCCESS;
}

uni_simd_result_e QpskCostas4Execute(
    uni_simd_qpsk_costas4_t& kernel,
    const uni_simd_qpsk_costas4_block_t& block,
    uni_simd_qpsk_costas4_state_t* const final_state) noexcept {
    if (block.descriptor_size != UNI_SIMD_QPSK_COSTAS4_BLOCK_DESCRIPTOR_SIZE ||
        (block.frequency_override_mask & ~0x0fU) != 0U ||
        block.sample_count > std::numeric_limits<std::size_t>::max() / (2U * sizeof(float))) {
        return UNI_SIMD_RESULT_INVALID_ARGUMENT;
    }
    const std::size_t channel_bytes = block.sample_count * 2U * sizeof(float);
    for (std::size_t lane = 0U; lane < UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT; ++lane) {
        if ((block.sample_count != 0U && block.channels[lane] == nullptr) ||
            (block.sample_count != 0U && !NaturallyAligned(block.channels[lane])) ||
            !std::isfinite(block.frequency_limit[lane]) || block.frequency_limit[lane] < 0.0f ||
            !std::isfinite(block.input_gain[lane]) || block.input_gain[lane] <= 0.0f ||
            !std::isfinite(block.frequency_override[lane])) {
            return UNI_SIMD_RESULT_INVALID_ARGUMENT;
        }
        for (std::size_t previous = 0U; previous < lane; ++previous) {
            if (Overlaps(block.channels[previous], channel_bytes, block.channels[lane], channel_bytes)) {
                return UNI_SIMD_RESULT_OVERLAPPING_BUFFERS;
            }
        }
        if (final_state != nullptr &&
            Overlaps(block.channels[lane], channel_bytes, final_state, sizeof(*final_state))) {
            return UNI_SIMD_RESULT_OVERLAPPING_BUFFERS;
        }
    }
    if (block.sample_count != 0U) {
        for (std::size_t lane = 0U; lane < UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT; ++lane) {
            if ((block.frequency_override_mask & (1U << lane)) != 0U) {
                kernel.state.frequency[lane] = block.frequency_override[lane];
            }
        }
        kernel.process(kernel, block);
    }
    if (final_state != nullptr) {
        *final_state = kernel.state;
    }
    return UNI_SIMD_RESULT_SUCCESS;
}

uni_simd_result_e QpskCostas4CopyState(uni_simd_qpsk_costas4_t& kernel, const uni_simd_qpsk_costas4_t& source) noexcept {
    if (std::memcmp(&kernel.config, &source.config, sizeof(kernel.config)) != 0) {
        return UNI_SIMD_RESULT_INVALID_ARGUMENT;
    }
    kernel.state = source.state;
    kernel.samples_since_normalization = source.samples_since_normalization;
    return UNI_SIMD_RESULT_SUCCESS;
}

uni_simd_result_e QpskCostas4Skip(uni_simd_qpsk_costas4_t& kernel, const std::size_t sample_count) noexcept {
    constexpr std::size_t delay = UNI_SIMD_QPSK_COSTAS4_LOOP_DELAY;
    auto& state = kernel.state;
    const std::size_t applied = std::min(sample_count, delay);
    for (std::size_t lane = 0U; lane < UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT; ++lane) {
        // The steps already decided take effect first; every later sample, with no detector
        // error, steps by the loop frequency alone. Summed in double: the gap can be long.
        double phase = state.phase[lane];
        for (std::size_t step = 0U; step < applied; ++step) {
            phase += state.pending_step[step][lane];
        }
        const float frequency = state.frequency[lane];
        phase += static_cast<double>(sample_count - applied) * static_cast<double>(frequency);
        state.phase[lane] = static_cast<float>(std::remainder(phase, 2.0 * std::numbers::pi));
        state.phase_cos[lane] = std::cos(state.phase[lane]);
        state.phase_sin[lane] = std::sin(state.phase[lane]);
        // The queue moves on by `applied` places and fills with frequency steps.
        for (std::size_t slot = 0U; slot < delay; ++slot) {
            if (slot + applied < delay) {
                state.pending_step[slot][lane] = state.pending_step[slot + applied][lane];
                state.pending_step_cos[slot][lane] = state.pending_step_cos[slot + applied][lane];
                state.pending_step_sin[slot][lane] = state.pending_step_sin[slot + applied][lane];
            } else {
                state.pending_step[slot][lane] = frequency;
                Costas4SinCos(frequency, state.pending_step_sin[slot][lane], state.pending_step_cos[slot][lane]);
            }
        }
        state.last_error[lane] = 0.0f;
    }
    return UNI_SIMD_RESULT_SUCCESS;
}

uni_simd_result_e QpskCostas4Rotate(uni_simd_qpsk_costas4_t& kernel, const std::int32_t* const quarter_turns) noexcept {
    if (quarter_turns == nullptr) {
        return UNI_SIMD_RESULT_INVALID_ARGUMENT;
    }
    auto& state = kernel.state;
    for (std::size_t lane = 0U; lane < UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT; ++lane) {
        // Multiplying the phasor by i^turns is exact: it swaps and negates components.
        const auto turns = static_cast<std::uint32_t>(quarter_turns[lane]) & 3U;
        for (std::uint32_t turn = 0U; turn < turns; ++turn) {
            const float phase_cos = state.phase_cos[lane];
            state.phase_cos[lane] = -state.phase_sin[lane];
            state.phase_sin[lane] = phase_cos;
        }
        state.phase[lane] = std::remainder(state.phase[lane] + static_cast<float>(turns) * static_cast<float>(std::numbers::pi / 2.0),
                                           2.0f * std::numbers::pi_v<float>);
    }
    return UNI_SIMD_RESULT_SUCCESS;
}

} // namespace uni::simd::kernels
