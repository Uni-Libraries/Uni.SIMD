#include <uni/simd/pfb_channelizer.hpp>

#include "detail/pfb_channelizer_internal.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>

namespace uni::simd {
namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

[[nodiscard]] constexpr bool supported_bin_count(const std::size_t count) noexcept {
    return count == 4U || count == 8U || count == 16U || count == 32U;
}

[[nodiscard]] constexpr std::size_t next_power_of_two(std::size_t value) noexcept {
    std::size_t result = 1U;
    while (result < value) {
        result *= 2U;
    }
    return result;
}

[[nodiscard]] float snapped_trigonometric(const double value) noexcept {
    constexpr double tolerance = 1.0e-12;
    if (std::abs(value) < tolerance) {
        return 0.0f;
    }
    if (std::abs(value - 1.0) < tolerance) {
        return 1.0f;
    }
    if (std::abs(value + 1.0) < tolerance) {
        return -1.0f;
    }
    return static_cast<float>(value);
}

void hash_word(std::uint64_t& hash, const std::uint64_t value) noexcept {
    constexpr std::uint64_t prime = 1099511628211ULL;
    for (std::size_t byte = 0U; byte < sizeof(value); ++byte) {
        hash ^= (value >> (byte * 8U)) & 0xffU;
        hash *= prime;
    }
}

} // namespace

Result make_pfb_channelizer_plan(PfbChannelizerPlan& destination,
                                 const PfbChannelizerPlanConfig& config) noexcept {
    if (!supported_bin_count(config.bin_count) || config.decimation == 0U ||
        config.bin_count % config.decimation != 0U || config.taps.empty() ||
        config.taps.size() > pfb_channelizer_max_taps ||
        config.logical_bins.size() > std::min(config.bin_count, pfb_channelizer_max_outputs) ||
        (config.grid_offset != PfbGridOffset::integer_bins && config.grid_offset != PfbGridOffset::half_bins)) {
        return Result::invalid_argument;
    }

    for (const float tap : config.taps) {
        if (!std::isfinite(tap)) {
            return Result::invalid_argument;
        }
    }

    const std::int32_t lowest_bin = -static_cast<std::int32_t>(config.bin_count / 2U);
    const std::int32_t highest_bin = static_cast<std::int32_t>(config.bin_count / 2U) - 1;
    for (std::size_t index = 0U; index < config.logical_bins.size(); ++index) {
        const std::int32_t logical_bin = config.logical_bins[index];
        if (logical_bin < lowest_bin || logical_bin > highest_bin) {
            return Result::invalid_argument;
        }
        for (std::size_t previous = 0U; previous < index; ++previous) {
            if (config.logical_bins[previous] == logical_bin) {
                return Result::invalid_argument;
            }
        }
    }

    PfbChannelizerPlan candidate{};
    candidate.bin_count_ = static_cast<std::uint16_t>(config.bin_count);
    candidate.decimation_ = static_cast<std::uint16_t>(config.decimation);
    candidate.tap_count_ = static_cast<std::uint16_t>(config.taps.size());
    candidate.row_count_ = static_cast<std::uint16_t>((config.taps.size() + config.bin_count - 1U) / config.bin_count);
    candidate.history_size_ = static_cast<std::uint16_t>(next_power_of_two(candidate.row_count_ * config.bin_count));
    candidate.selected_count_ = static_cast<std::uint8_t>(config.logical_bins.size());
    candidate.grid_offset_ = config.grid_offset;
    std::copy(config.logical_bins.begin(), config.logical_bins.end(), candidate.logical_bins_.begin());

    const bool half_grid = config.grid_offset == PfbGridOffset::half_bins;
    for (std::size_t row = 0U; row < candidate.row_count_; ++row) {
        const float row_sign = half_grid && row % 2U != 0U ? -1.0f : 1.0f;
        for (std::size_t branch = 0U; branch < config.bin_count; ++branch) {
            const std::size_t tap_index = row * config.bin_count + branch;
            candidate.coefficients_[tap_index] =
                tap_index < config.taps.size() ? row_sign * config.taps[tap_index] : 0.0f;
        }
        for (std::size_t branch = 0U; branch < config.bin_count; ++branch) {
            candidate.reversed_coefficients_[row * config.bin_count + branch] =
                candidate.coefficients_[row * config.bin_count + config.bin_count - 1U - branch];
        }
    }

    const double delta = half_grid ? 0.5 : 0.0;
    for (std::size_t branch = 0U; branch < config.bin_count; ++branch) {
        const double angle = 2.0 * pi * delta * static_cast<double>(branch) / static_cast<double>(config.bin_count);
        candidate.branch_rotation_re_[branch] = snapped_trigonometric(std::cos(angle));
        candidate.branch_rotation_im_[branch] = snapped_trigonometric(std::sin(angle));
    }
    for (std::size_t twiddle = 0U; twiddle < config.bin_count / 2U; ++twiddle) {
        const double angle = 2.0 * pi * static_cast<double>(twiddle) / static_cast<double>(config.bin_count);
        candidate.fft_twiddle_re_[twiddle] = snapped_trigonometric(std::cos(angle));
        candidate.fft_twiddle_im_[twiddle] = snapped_trigonometric(std::sin(angle));
    }

    std::size_t bit_count = 0U;
    for (std::size_t value = config.bin_count; value > 1U; value /= 2U) {
        ++bit_count;
    }
    for (std::size_t value = 0U; value < config.bin_count; ++value) {
        std::size_t source = value;
        std::size_t reversed = 0U;
        for (std::size_t bit = 0U; bit < bit_count; ++bit) {
            reversed = (reversed << 1U) | (source & 1U);
            source >>= 1U;
        }
        candidate.bit_reverse_[value] = static_cast<std::uint8_t>(reversed);
    }

    std::size_t phase_period = 1U;
    const std::int64_t phase_modulus = static_cast<std::int64_t>(half_grid ? 2U * config.bin_count : config.bin_count);
    for (const std::int32_t logical_bin : config.logical_bins) {
        const std::int64_t bin_numerator = half_grid ? 2LL * logical_bin + 1LL : logical_bin;
        const std::int64_t phase_step = std::abs(bin_numerator * static_cast<std::int64_t>(config.decimation));
        const std::size_t output_period =
            static_cast<std::size_t>(phase_modulus / std::gcd(phase_modulus, phase_step));
        phase_period = std::lcm(phase_period, output_period);
    }
    if (phase_period > pfb_channelizer_max_phase_period) {
        return Result::invalid_argument;
    }
    candidate.phase_period_ = static_cast<std::uint8_t>(phase_period);

    for (std::size_t output = 0U; output < config.logical_bins.size(); ++output) {
        const double frequency_bin = static_cast<double>(config.logical_bins[output]) + delta;
        for (std::size_t phase = 0U; phase < phase_period; ++phase) {
            const double angle = -2.0 * pi * frequency_bin * static_cast<double>(config.decimation * phase) /
                                 static_cast<double>(config.bin_count);
            const std::size_t table_index = output * pfb_channelizer_max_phase_period + phase;
            candidate.post_phase_re_[table_index] = snapped_trigonometric(std::cos(angle));
            candidate.post_phase_im_[table_index] = snapped_trigonometric(std::sin(angle));
        }
    }

    std::uint64_t signature = 1469598103934665603ULL;
    hash_word(signature, config.bin_count);
    hash_word(signature, config.decimation);
    hash_word(signature, static_cast<std::uint8_t>(config.grid_offset));
    hash_word(signature, config.taps.size());
    for (const float tap : config.taps) {
        hash_word(signature, std::bit_cast<std::uint32_t>(tap));
    }
    for (const std::int32_t logical_bin : config.logical_bins) {
        hash_word(signature, static_cast<std::uint32_t>(logical_bin));
    }
    candidate.signature_ = signature == 0U ? 1U : signature;

    destination = std::move(candidate);
    return Result::success;
}

Result reset_pfb_channelizer_state(PfbChannelizerState& state, const PfbChannelizerPlan& plan) noexcept {
    if (!detail::pfb_plan_is_valid(plan)) {
        return Result::invalid_argument;
    }
    PfbChannelizerState candidate{};
    candidate.plan_signature_ = plan.signature_;
    state = std::move(candidate);
    return Result::success;
}

std::expected<std::size_t, Result> pfb_channelizer_output_count(const PfbChannelizerPlan& plan,
                                                               const PfbChannelizerState& state,
                                                               const std::size_t input_count) noexcept {
    if (!detail::pfb_plan_is_valid(plan) || !detail::pfb_state_is_valid(plan, state)) {
        return std::unexpected(Result::invalid_argument);
    }
    return detail::pfb_output_count_unchecked(plan, state, input_count);
}

} // namespace uni::simd
