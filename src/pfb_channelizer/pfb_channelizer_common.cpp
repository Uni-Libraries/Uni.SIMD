#include "common/api_internal.hpp"

#include "pfb_channelizer/pfb_channelizer_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <numeric>

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

template <typename Left, typename Right>
[[nodiscard]] bool overlaps(const std::span<Left> left, const std::span<Right> right) noexcept {
    if (left.empty() || right.empty()) {
        return false;
    }
    const auto left_begin = reinterpret_cast<std::uintptr_t>(left.data());
    const auto right_begin = reinterpret_cast<std::uintptr_t>(right.data());
    return left_begin <= right_begin ? right_begin - left_begin < left.size_bytes()
                                     : left_begin - right_begin < right.size_bytes();
}

} // namespace

namespace detail {

bool PfbChannelizer_supports_all(const PfbChannelizerData&) noexcept {
    return true;
}

std::expected<std::unique_ptr<PfbChannelizerData>, Result>
make_pfb_channelizer_data(const PfbChannelizerConfig& config, const PfbChannelizerFn candidate,
                           const PfbChannelizerSupportFn supports, const Backend backend,
                           const PfbChannelizerFn fallback,
                           const PfbChannelizerSupportFn fallback_supports,
                           const Backend fallback_backend) noexcept {
    if (!supported_bin_count(config.bin_count) || config.decimation == 0U ||
        config.bin_count % config.decimation != 0U || config.taps.empty() ||
        config.taps.size() > pfb_channelizer_max_taps ||
        config.logical_bins.size() > std::min(config.bin_count, pfb_channelizer_max_outputs) ||
        (config.grid_offset != PfbGridOffset::integer_bins && config.grid_offset != PfbGridOffset::half_bins)) {
        return std::unexpected(Result::invalid_argument);
    }
    if (std::any_of(config.taps.begin(), config.taps.end(), [](const float tap) { return !std::isfinite(tap); })) {
        return std::unexpected(Result::invalid_argument);
    }

    const std::int32_t lowest_bin = -static_cast<std::int32_t>(config.bin_count / 2U);
    const std::int32_t highest_bin = static_cast<std::int32_t>(config.bin_count / 2U) - 1;
    for (std::size_t index = 0U; index < config.logical_bins.size(); ++index) {
        const auto bin = config.logical_bins[index];
        if (bin < lowest_bin || bin > highest_bin ||
            std::find(config.logical_bins.begin(), config.logical_bins.begin() + static_cast<std::ptrdiff_t>(index), bin) !=
                config.logical_bins.begin() + static_cast<std::ptrdiff_t>(index)) {
            return std::unexpected(Result::invalid_argument);
        }
    }

    try {
        auto data = std::make_unique<PfbChannelizerData>();
        data->bins = config.bin_count;
        data->decimation_value = config.decimation;
        data->taps = config.taps.size();
        data->rows = (config.taps.size() + config.bin_count - 1U) / config.bin_count;
        const std::size_t filter_span = data->rows * config.bin_count;
        data->history_size = next_power_of_two(filter_span + 3U * config.decimation);
        data->offset = config.grid_offset;
        data->selected_bins.assign(config.logical_bins.begin(), config.logical_bins.end());
        data->selected_fft_bins.resize(data->selected_bins.size());
        for (std::size_t output = 0U; output < data->selected_bins.size(); ++output) {
            data->selected_fft_bins[output] = static_cast<std::size_t>(
                data->selected_bins[output] < 0
                    ? data->selected_bins[output] + static_cast<std::int32_t>(data->bins)
                    : data->selected_bins[output]);
        }
        data->reversed_coefficients.resize(data->rows * data->bins);

        const bool half_grid = config.grid_offset == PfbGridOffset::half_bins;
        for (std::size_t row = 0U; row < data->rows; ++row) {
            const float row_sign = half_grid && row % 2U != 0U ? -1.0f : 1.0f;
            for (std::size_t branch = 0U; branch < data->bins; ++branch) {
                const std::size_t tap_index = row * data->bins + branch;
                data->reversed_coefficients[row * data->bins + data->bins - 1U - branch] =
                    tap_index < config.taps.size() ? row_sign * config.taps[tap_index] : 0.0f;
            }
        }

        const double delta = half_grid ? 0.5 : 0.0;
        for (std::size_t branch = 0U; branch < data->bins; ++branch) {
            const double angle = 2.0 * pi * delta * static_cast<double>(branch) / static_cast<double>(data->bins);
            data->branch_rotation_re[branch] = snapped_trigonometric(std::cos(angle));
            data->branch_rotation_im[branch] = snapped_trigonometric(std::sin(angle));
        }

        std::size_t phase_period = 1U;
        const std::int64_t modulus = static_cast<std::int64_t>(half_grid ? 2U * data->bins : data->bins);
        for (const std::int32_t logical_bin : data->selected_bins) {
            const std::int64_t numerator = half_grid ? 2LL * logical_bin + 1LL : logical_bin;
            const std::int64_t step = std::abs(numerator * static_cast<std::int64_t>(data->decimation_value));
            phase_period = std::lcm(phase_period, static_cast<std::size_t>(modulus / std::gcd(modulus, step)));
        }
        data->phase_period = phase_period;
        data->post_phase_re.resize(data->selected_bins.size() * phase_period);
        data->post_phase_im.resize(data->selected_bins.size() * phase_period);
        if (data->selected_bins.size() == 1U) {
            data->selected_transform_re.resize(data->bins);
            data->selected_transform_im.resize(data->bins);
        }
        for (std::size_t output = 0U; output < data->selected_bins.size(); ++output) {
            const double frequency_bin = static_cast<double>(data->selected_bins[output]) + delta;
            for (std::size_t phase = 0U; phase < phase_period; ++phase) {
                const double angle = -2.0 * pi * frequency_bin * static_cast<double>(data->decimation_value * phase) /
                                     static_cast<double>(data->bins);
                const std::size_t table_index = output * phase_period + phase;
                data->post_phase_re[table_index] = snapped_trigonometric(std::cos(angle));
                data->post_phase_im[table_index] = snapped_trigonometric(std::sin(angle));
            }
            if (data->selected_bins.size() == 1U) {
                const std::size_t fft_bin = static_cast<std::size_t>(
                    data->selected_bins[output] < 0
                        ? data->selected_bins[output] + static_cast<std::int32_t>(data->bins)
                        : data->selected_bins[output]);
                for (std::size_t branch = 0U; branch < data->bins; ++branch) {
                    const double angle = 2.0 * pi * static_cast<double>(branch * fft_bin) /
                                         static_cast<double>(data->bins);
                    const double weight_re = std::cos(angle);
                    const double weight_im = std::sin(angle);
                    const double rotation_re = data->branch_rotation_re[branch];
                    const double rotation_im = data->branch_rotation_im[branch];
                    data->selected_transform_re[branch] = snapped_trigonometric(
                        rotation_re * weight_re - rotation_im * weight_im);
                    data->selected_transform_im[branch] = snapped_trigonometric(
                        rotation_re * weight_im + rotation_im * weight_re);
                }
            }
        }
        data->history.resize(4U * data->history_size);
        if (candidate != nullptr && supports != nullptr && supports(*data)) {
            data->process = candidate;
            data->selected_backend = backend;
        } else if (fallback != nullptr && fallback_supports != nullptr && fallback_supports(*data)) {
            data->process = fallback;
            data->selected_backend = fallback_backend;
        } else {
            data->process = &PfbChannelizer_generic;
            data->selected_backend = Backend::generic;
        }
        return data;
    } catch (const std::bad_alloc&) {
        return std::unexpected(Result::out_of_memory);
    }
}

} // namespace detail

PfbChannelizer::PfbChannelizer() noexcept = default;
PfbChannelizer::~PfbChannelizer() = default;
PfbChannelizer::PfbChannelizer(PfbChannelizer&&) noexcept = default;
PfbChannelizer& PfbChannelizer::operator=(PfbChannelizer&&) noexcept = default;

bool PfbChannelizer::initialized() const noexcept { return data_ != nullptr; }
Backend PfbChannelizer::backend() const noexcept { return data_ ? data_->selected_backend : Backend::generic; }
std::size_t PfbChannelizer::bin_count() const noexcept { return data_ ? data_->bins : 0U; }
std::size_t PfbChannelizer::decimation() const noexcept { return data_ ? data_->decimation_value : 0U; }
std::size_t PfbChannelizer::tap_count() const noexcept { return data_ ? data_->taps : 0U; }
std::span<const std::int32_t> PfbChannelizer::logical_bins() const noexcept {
    return data_ ? std::span<const std::int32_t>{data_->selected_bins} : std::span<const std::int32_t>{};
}

Result PfbChannelizer::reset() noexcept {
    if (!data_) {
        return Result::invalid_argument;
    }
    std::fill(data_->history.begin(), data_->history.end(), 0.0f);
    data_->cursor = 0U;
    data_->decimation_phase = 0U;
    data_->post_phase = 0U;
    return Result::success;
}

std::expected<std::size_t, Result> PfbChannelizer::output_count(const std::size_t input_count) const noexcept {
    if (!data_) {
        return std::unexpected(Result::invalid_argument);
    }
    return detail::pfb_output_count_unchecked(*data_, input_count);
}

std::expected<std::size_t, Result> PfbChannelizer::process(const PfbChannelizerBlock& block) noexcept {
    if (!data_) {
        return std::unexpected(Result::invalid_argument);
    }
    if (block.input.size() % 2U != 0U) {
        return std::unexpected(Result::invalid_size);
    }
    const std::size_t expected = detail::pfb_output_count_unchecked(*data_, block.input.size() / 2U);
    for (std::size_t output = 0U; output < data_->selected_bins.size(); ++output) {
        if (block.outputs[output].size() < expected * 2U) {
            return std::unexpected(Result::invalid_size);
        }
        const auto active = block.outputs[output].first(expected * 2U);
        if (overlaps(active, block.input)) {
            return std::unexpected(Result::overlapping_buffers);
        }
        for (std::size_t previous = 0U; previous < output; ++previous) {
            if (overlaps(active, block.outputs[previous].first(expected * 2U))) {
                return std::unexpected(Result::overlapping_buffers);
            }
        }
    }
    return data_->process(*data_, block);
}

} // namespace uni::simd
