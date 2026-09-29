#include "symbol_sync_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace uni::simd::kernels {

namespace {

[[nodiscard]] bool ValidBackend(const uni_simd_backend_e backend) noexcept { return backend <= UNI_SIMD_BACKEND_AARCH64_NEON; }

[[nodiscard]] bool FiniteNonNegative(const double value) noexcept { return std::isfinite(value) && value >= 0.0; }

[[nodiscard]] bool ValidConfig(const uni_simd_symbol_sync4_config_t& config) noexcept {
    if (config.descriptor_size != UNI_SIMD_SYMBOL_SYNC4_CONFIG_DESCRIPTOR_SIZE ||
        (config.interpolator != UNI_SIMD_SYMBOL_SYNC_INTERPOLATOR_CUBIC && config.interpolator != UNI_SIMD_SYMBOL_SYNC_INTERPOLATOR_LINEAR) ||
        (config.ted != UNI_SIMD_SYMBOL_SYNC_TED_GARDNER && config.ted != UNI_SIMD_SYMBOL_SYNC_TED_MUELLER_MULLER)) {
        return false;
    }
    for (std::size_t lane = 0U; lane < kSymbolSync4Lanes; ++lane) {
        // Negated comparisons also reject NaN.
        if (!(config.omega_min[lane] >= 1.0) || !(config.omega[lane] >= config.omega_min[lane]) || !(config.omega_max[lane] >= config.omega[lane]) ||
            !(config.omega_max[lane] <= UNI_SIMD_SYMBOL_SYNC4_MAX_OMEGA) || !FiniteNonNegative(config.alpha[lane]) || !FiniteNonNegative(config.beta[lane])) {
            return false;
        }
    }
    if (config.auto_ted != 0U && config.ted == UNI_SIMD_SYMBOL_SYNC_TED_GARDNER) {
        return config.lock_ema_alpha > 0.0 && config.lock_ema_alpha < 1.0 && FiniteNonNegative(config.mm_threshold) &&
               FiniteNonNegative(config.fallback_threshold);
    }
    return true;
}

[[nodiscard]] bool NaturallyAligned(const void* const pointer) noexcept { return reinterpret_cast<std::uintptr_t>(pointer) % alignof(float) == 0U; }

[[nodiscard]] bool Overlaps(const void* const left, const std::size_t left_bytes, const void* const right, const std::size_t right_bytes) noexcept {
    if (left_bytes == 0U || right_bytes == 0U) {
        return false;
    }
    const auto left_begin = reinterpret_cast<std::uintptr_t>(left);
    const auto right_begin = reinterpret_cast<std::uintptr_t>(right);
    return left_begin <= right_begin ? right_begin - left_begin < left_bytes : left_begin - right_begin < right_bytes;
}

[[nodiscard]] double HoldCount(const std::uint32_t hold) noexcept {
    return hold == 0U ? std::numeric_limits<double>::infinity() : static_cast<double>(hold);
}

[[nodiscard]] double HalfSymbolCeiling(const uni_simd_symbol_sync4_config_t& config) noexcept {
    return std::ceil(*std::max_element(std::begin(config.omega_max), std::end(config.omega_max)) * 0.5);
}

/**
 * Earliest position of a symbol at the start of a call (relative to the new block) whose
 * interpolation, Gardner mid-point included, still fits into the kept history.
 */
[[nodiscard]] double EarliestTime(const uni_simd_symbol_sync4_t& kernel) noexcept {
    return -(static_cast<double>(kernel.history_length) - HalfSymbolCeiling(kernel.config) - 4.0);
}

} // namespace

void SymbolSync4SwitchTed(SymbolSync4State& state, const std::size_t lane) noexcept {
    state.mueller_muller[lane] ^= 1U;
    state.err_norm2_ema[lane] = 0.0;
    state.lock_run[lane] = 0.0;
    state.unlock_run[lane] = 0.0;
    ++state.ted_switches[lane];
}

uni_simd_result_e SymbolSync4Initialize(uni_simd_symbol_sync4_t& kernel, const uni_simd_symbol_sync4_config_t& config,
                                        const uni_simd_backend_e requested_backend, const uni_simd_math_mode_e math_mode) noexcept {
    if (!ValidBackend(requested_backend) || math_mode > UNI_SIMD_MATH_DETERMINISTIC || !ValidConfig(config)) {
        return UNI_SIMD_RESULT_INVALID_ARGUMENT;
    }
    kernel = {};
    kernel.config = config;

    auto& params = kernel.params;
    for (std::size_t lane = 0U; lane < kSymbolSync4Lanes; ++lane) {
        params.omega_min[lane] = config.omega_min[lane];
        params.omega_max[lane] = config.omega_max[lane];
        params.alpha[lane] = config.alpha[lane];
        params.beta[lane] = config.beta[lane];
        params.alpha_beta[lane] = config.alpha[lane] + config.beta[lane];
    }
    params.cubic = config.interpolator == UNI_SIMD_SYMBOL_SYNC_INTERPOLATOR_CUBIC;
    params.auto_ted = config.auto_ted != 0U && config.ted == UNI_SIMD_SYMBOL_SYNC_TED_GARDNER;
    params.auto_ted_fallback = params.auto_ted && config.auto_ted_fallback != 0U;
    params.lock_ema_alpha = config.lock_ema_alpha;
    params.mm_min_symbols = static_cast<double>(config.mm_min_symbols);
    params.mm_hold_symbols = HoldCount(config.mm_hold_symbols);
    params.fallback_hold_symbols = HoldCount(config.fallback_hold_symbols);
    params.mm_threshold = config.mm_threshold;
    params.fallback_threshold = config.fallback_threshold;

    const auto history = static_cast<std::size_t>(HalfSymbolCeiling(config)) + 8U;
    kernel.history_length = (history + 3U) & ~std::size_t{3U};

    auto& initial = kernel.initial_state;
    for (std::size_t lane = 0U; lane < kSymbolSync4Lanes; ++lane) {
        // The first symbol sits one sample into the stream, where the cubic stencil starts.
        initial.time[lane] = 1.0;
        initial.omega[lane] = config.omega[lane];
        initial.mueller_muller[lane] = config.ted == UNI_SIMD_SYMBOL_SYNC_TED_MUELLER_MULLER ? 1U : 0U;
    }
    kernel.state = initial;

    kernel.run = &SymbolSync4_generic;
    kernel.backend = UNI_SIMD_BACKEND_GENERIC;
    const bool deterministic = math_mode == UNI_SIMD_MATH_DETERMINISTIC;
#if UNI_SIMD_HAVE_AVX2_FMA
    const auto& capabilities = uni::simd::capabilities();
    const bool avx2_requested = requested_backend == UNI_SIMD_BACKEND_AUTOMATIC || requested_backend == UNI_SIMD_BACKEND_X86_AVX2_FMA ||
                                requested_backend == UNI_SIMD_BACKEND_X86_AVX512;
    if (!deterministic && capabilities.avx2 && capabilities.fma && avx2_requested) {
        kernel.run = &SymbolSync4_avx2;
        kernel.backend = UNI_SIMD_BACKEND_X86_AVX2_FMA;
    }
#endif
    if (requested_backend != UNI_SIMD_BACKEND_AUTOMATIC && requested_backend != UNI_SIMD_BACKEND_GENERIC && kernel.backend == UNI_SIMD_BACKEND_GENERIC &&
        !deterministic) {
        return UNI_SIMD_RESULT_UNSUPPORTED_BACKEND;
    }
    return UNI_SIMD_RESULT_SUCCESS;
}

uni_simd_result_e SymbolSync4Reset(uni_simd_symbol_sync4_t& kernel) noexcept {
    kernel.state = kernel.initial_state;
    std::memset(kernel.history, 0, sizeof(kernel.history));
    return UNI_SIMD_RESULT_SUCCESS;
}

/**
 * Runs the timing loops over one block per lane.
 *
 * A symbol needs the samples around it and, for Gardner, around the point half a symbol
 * earlier, so the first symbols of a block also read the end of the previous one. Rather than
 * making the inner loops check which buffer a sample is in, every call runs two passes:
 *   1. the seam: `history_length` kept samples followed by the first `history_length` samples
 *      of the block, copied into one small buffer per lane;
 *   2. the block itself, once every lane's next symbol lies far enough into it that nothing
 *      before the block start is needed.
 * A lane stops a pass at the first symbol whose stencil would run past the end of the data;
 * that symbol is produced by the next pass or call.
 */
uni_simd_result_e SymbolSync4Execute(uni_simd_symbol_sync4_t& kernel, const uni_simd_symbol_sync4_block_t& block,
                                     uni_simd_symbol_sync4_result_t& result) noexcept {
    if (block.descriptor_size != UNI_SIMD_SYMBOL_SYNC4_BLOCK_DESCRIPTOR_SIZE || result.descriptor_size != UNI_SIMD_SYMBOL_SYNC4_RESULT_DESCRIPTOR_SIZE) {
        return UNI_SIMD_RESULT_INVALID_ARGUMENT;
    }
    constexpr std::size_t sample_bytes = 2U * sizeof(float);
    constexpr std::size_t max_count = std::numeric_limits<std::int32_t>::max() / 2;
    for (std::size_t lane = 0U; lane < kSymbolSync4Lanes; ++lane) {
        const std::size_t input_count = block.input_count[lane];
        const std::size_t output_capacity = block.output_capacity[lane];
        if (input_count > max_count || output_capacity > max_count) {
            return UNI_SIMD_RESULT_INVALID_SIZE;
        }
        if ((input_count != 0U && (block.input[lane] == nullptr || !NaturallyAligned(block.input[lane]))) ||
            (output_capacity != 0U && (block.output[lane] == nullptr || !NaturallyAligned(block.output[lane])))) {
            return UNI_SIMD_RESULT_INVALID_ARGUMENT;
        }
        for (std::size_t other = 0U; other < kSymbolSync4Lanes; ++other) {
            if (Overlaps(block.output[lane], output_capacity * sample_bytes, block.input[other], block.input_count[other] * sample_bytes) ||
                (other < lane && Overlaps(block.output[lane], output_capacity * sample_bytes, block.output[other], block.output_capacity[other] * sample_bytes))) {
                return UNI_SIMD_RESULT_OVERLAPPING_BUFFERS;
            }
        }
    }

    auto& state = kernel.state;
    const std::size_t history = kernel.history_length;
    const auto history_time = static_cast<double>(history);
    SymbolSync4Segments segments{};

    // Pass 1: the seam.
    for (std::size_t lane = 0U; lane < kSymbolSync4Lanes; ++lane) {
        const std::size_t head = std::min(block.input_count[lane], history);
        float* const seam = kernel.seam[lane];
        std::memcpy(seam, kernel.history[lane], history * sample_bytes);
        if (head != 0U) {
            std::memcpy(seam + 2U * history, block.input[lane], head * sample_bytes);
        }
        segments.data[lane] = seam;
        segments.length[lane] = history + head;
        segments.limit[lane] = static_cast<double>(history + head) - 2.0;
        segments.output[lane] = block.output[lane];
        segments.output_capacity[lane] = block.output_capacity[lane];
        state.time[lane] += history_time;
    }
    kernel.run(kernel.params, state, segments);

    // Pass 2: the rest of the block. Lanes whose block fitted into the seam stay idle.
    for (std::size_t lane = 0U; lane < kSymbolSync4Lanes; ++lane) {
        state.time[lane] -= history_time;
        const std::size_t count = block.input_count[lane];
        if (count > history) {
            segments.data[lane] = block.input[lane];
            segments.length[lane] = count;
            segments.limit[lane] = static_cast<double>(count) - 2.0;
        } else {
            segments.limit[lane] = -std::numeric_limits<double>::infinity();
        }
    }
    kernel.run(kernel.params, state, segments);

    // Keep the tail for the next seam and move the time origin to the next block.
    const double earliest = EarliestTime(kernel);
    result.truncated_mask = 0U;
    for (std::size_t lane = 0U; lane < kSymbolSync4Lanes; ++lane) {
        const std::size_t count = block.input_count[lane];
        const float* const tail = count >= history ? block.input[lane] + 2U * (count - history) : kernel.seam[lane] + 2U * count;
        std::memmove(kernel.history[lane], tail, history * sample_bytes);

        if (segments.output_count[lane] == segments.output_capacity[lane] && state.time[lane] < static_cast<double>(count) - 2.0) {
            result.truncated_mask |= 1U << lane;
        }
        state.time[lane] -= static_cast<double>(count);
        if (!std::isfinite(state.time[lane]) || !std::isfinite(state.omega[lane])) {
            // A non-finite input sample poisoned the loop; restart this lane's timing.
            state.time[lane] = earliest;
            state.omega[lane] = kernel.config.omega[lane];
            state.have_previous[lane] = 0U;
            state.err_norm2_ema[lane] = 0.0;
        }
        // A truncated lane (or a loop that stepped backwards) would otherwise need samples
        // that are no longer kept; skip ahead instead.
        state.time[lane] = std::max(state.time[lane], earliest);

        result.output_count[lane] = segments.output_count[lane];
        result.ted[lane] = state.mueller_muller[lane] != 0U ? UNI_SIMD_SYMBOL_SYNC_TED_MUELLER_MULLER : UNI_SIMD_SYMBOL_SYNC_TED_GARDNER;
        result.omega[lane] = state.omega[lane];
        result.err_norm2_ema[lane] = state.err_norm2_ema[lane];
        result.symbols[lane] = static_cast<std::uint64_t>(state.symbols[lane]);
        result.ted_switches[lane] = state.ted_switches[lane];
    }
    return UNI_SIMD_RESULT_SUCCESS;
}

} // namespace uni::simd::kernels
