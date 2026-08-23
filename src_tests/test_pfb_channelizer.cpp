#ifdef NDEBUG
#undef NDEBUG
#endif

#include "common/api_internal.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace {

using Complex = std::complex<float>;
constexpr double pi = 3.141592653589793238462643383279502884;

struct RunResult final {
    std::vector<std::vector<Complex>> outputs;
    std::size_t produced = 0U;
    uni::simd::Backend backend = uni::simd::Backend::generic;
};

[[nodiscard]] std::span<const float> as_components(const std::span<const Complex> values) noexcept {
    return {reinterpret_cast<const float*>(values.data()), values.size() * 2U};
}

[[nodiscard]] std::span<float> as_components(const std::span<Complex> values) noexcept {
    return {reinterpret_cast<float*>(values.data()), values.size() * 2U};
}

[[nodiscard]] std::vector<float> make_taps(const std::size_t count) {
    std::vector<float> taps(count);
    const float scale = 0.5f / std::sqrt(static_cast<float>(count));
    for (std::size_t index = 0U; index < count; ++index) {
        const float centered = static_cast<float>(static_cast<std::int64_t>(index) - static_cast<std::int64_t>(count / 2U));
        taps[index] = scale * (0.7f * std::cos(centered * 0.173f) + 0.3f * std::sin(centered * 0.071f));
    }
    return taps;
}

[[nodiscard]] std::vector<Complex> make_input(const std::size_t count) {
    std::vector<Complex> input(count);
    std::uint32_t random = 0x12345678U;
    for (auto& sample : input) {
        random = random * 1664525U + 1013904223U;
        const float real = static_cast<float>(static_cast<std::int32_t>(random)) / 2147483648.0f;
        random = random * 1664525U + 1013904223U;
        const float imag = static_cast<float>(static_cast<std::int32_t>(random)) / 2147483648.0f;
        sample = {real, imag};
    }
    return input;
}

[[nodiscard]] RunResult run(const uni::simd::Context& context, const uni::simd::PfbChannelizerConfig& config,
                            const std::span<const Complex> input,
                            const std::span<const std::size_t> split_pattern = {}) {
    auto channelizer_result = context.make_pfb_channelizer(config);
    assert(channelizer_result.has_value());
    auto channelizer = std::move(*channelizer_result);
    RunResult result;
    result.backend = channelizer.backend();
    const auto total = channelizer.output_count(input.size());
    assert(total.has_value());
    result.outputs.resize(config.logical_bins.size());
    for (auto& output : result.outputs) {
        output.resize(*total);
    }

    std::size_t input_offset = 0U;
    std::size_t split_index = 0U;
    bool called = false;
    do {
        const std::size_t requested = split_pattern.empty() ? input.size() : split_pattern[split_index++ % split_pattern.size()];
        const std::size_t count = std::min(requested, input.size() - input_offset);
        uni::simd::PfbChannelizerBlock block{
            .input = as_components(input.subspan(input_offset, count)),
        };
        for (std::size_t output = 0U; output < result.outputs.size(); ++output) {
            block.outputs[output] = as_components(
                std::span<Complex>{result.outputs[output]}.subspan(result.produced));
        }
        const auto produced = channelizer.process(block);
        assert(produced.has_value());
        result.produced += *produced;
        input_offset += count;
        called = true;
    } while (input_offset < input.size() || !called);
    assert(result.produced == *total);
    return result;
}

[[nodiscard]] std::vector<std::vector<std::complex<double>>>
direct_reference(const uni::simd::PfbChannelizerConfig& config, const std::span<const Complex> input) {
    const std::size_t count = input.empty() ? 0U : 1U + (input.size() - 1U) / config.decimation;
    std::vector<std::vector<std::complex<double>>> result(config.logical_bins.size(),
                                                          std::vector<std::complex<double>>(count));
    const double delta = config.grid_offset == uni::simd::PfbGridOffset::half_bins ? 0.5 : 0.0;
    for (std::size_t output = 0U; output < config.logical_bins.size(); ++output) {
        const double frequency_bin = static_cast<double>(config.logical_bins[output]) + delta;
        for (std::size_t hop = 0U; hop < count; ++hop) {
            const std::size_t sample_index = hop * config.decimation;
            std::complex<double> accumulator{};
            for (std::size_t tap = 0U; tap < std::min(config.taps.size(), sample_index + 1U); ++tap) {
                const std::size_t source_index = sample_index - tap;
                const double angle = -2.0 * pi * frequency_bin * static_cast<double>(source_index) /
                                     static_cast<double>(config.bin_count);
                accumulator += static_cast<double>(config.taps[tap]) *
                               std::complex<double>{input[source_index].real(), input[source_index].imag()} *
                               std::complex<double>{std::cos(angle), std::sin(angle)};
            }
            result[output][hop] = accumulator;
        }
    }
    return result;
}

void compare_reference(const RunResult& actual,
                       const std::vector<std::vector<std::complex<double>>>& expected,
                       const float absolute_tolerance = 3.0e-5f,
                       const float relative_tolerance = 4.0e-4f) {
    assert(actual.outputs.size() == expected.size());
    for (std::size_t output = 0U; output < expected.size(); ++output) {
        assert(actual.outputs[output].size() == expected[output].size());
        for (std::size_t index = 0U; index < expected[output].size(); ++index) {
            const Complex reference{static_cast<float>(expected[output][index].real()),
                                    static_cast<float>(expected[output][index].imag())};
            const float tolerance = absolute_tolerance + relative_tolerance * std::abs(reference);
            assert(std::isfinite(actual.outputs[output][index].real()));
            assert(std::isfinite(actual.outputs[output][index].imag()));
            assert(std::abs(actual.outputs[output][index] - reference) <= tolerance);
        }
    }
}

void compare_runs(const RunResult& reference, const RunResult& actual) {
    assert(reference.produced == actual.produced);
    assert(reference.outputs.size() == actual.outputs.size());
    for (std::size_t output = 0U; output < reference.outputs.size(); ++output) {
        for (std::size_t index = 0U; index < reference.outputs[output].size(); ++index) {
            const float tolerance = 3.0e-5f + 5.0e-4f * std::abs(reference.outputs[output][index]);
            assert(std::abs(reference.outputs[output][index] - actual.outputs[output][index]) <= tolerance);
        }
    }
}

void test_validation(const uni::simd::Context& generic) {
    const std::array<float, 1U> tap{1.0f};
    constexpr std::array<std::int32_t, 2U> bins{-1, 1};
    auto create = [&](const uni::simd::PfbChannelizerConfig& config) { return generic.make_pfb_channelizer(config); };
    assert(!create({.bin_count = 3U, .decimation = 1U, .taps = tap}).has_value());
    assert(!create({.bin_count = 8U, .decimation = 0U, .taps = tap}).has_value());
    assert(!create({.bin_count = 8U, .decimation = 3U, .taps = tap}).has_value());
    assert(!create({.bin_count = 8U, .decimation = 4U}).has_value());
    std::vector<float> too_many(uni::simd::pfb_channelizer_max_taps + 1U, 1.0f);
    assert(!create({.bin_count = 8U, .decimation = 4U, .taps = too_many}).has_value());
    for (const float bad : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN()}) {
        const std::array bad_tap{bad};
        assert(!create({.bin_count = 8U, .decimation = 4U, .taps = bad_tap}).has_value());
    }
    constexpr std::array duplicate{std::int32_t{0}, std::int32_t{0}};
    assert(!create({.bin_count = 8U, .decimation = 4U, .taps = tap, .logical_bins = duplicate}).has_value());
    constexpr std::array out_of_range{std::int32_t{4}};
    assert(!create({.bin_count = 8U, .decimation = 4U, .taps = tap, .logical_bins = out_of_range}).has_value());

    auto channelizer = create({.bin_count = 8U, .decimation = 4U, .grid_offset = uni::simd::PfbGridOffset::half_bins,
                               .taps = tap, .logical_bins = bins});
    assert(channelizer.has_value());
    const auto input = make_input(17U);
    std::array<Complex, 4U> short_output{};
    std::array<Complex, 5U> full_output{};
    uni::simd::PfbChannelizerBlock short_block{.input = as_components(std::span<const Complex>{input})};
    short_block.outputs[0] = as_components(std::span<Complex>{short_output});
    short_block.outputs[1] = as_components(std::span<Complex>{full_output});
    const auto short_result = channelizer->process(short_block);
    assert(!short_result && short_result.error() == uni::simd::Result::invalid_size);

    std::array<Complex, 5U> shared{};
    uni::simd::PfbChannelizerBlock overlap{.input = as_components(std::span<const Complex>{input})};
    overlap.outputs[0] = as_components(std::span<Complex>{shared});
    overlap.outputs[1] = as_components(std::span<Complex>{shared});
    const auto overlap_result = channelizer->process(overlap);
    assert(!overlap_result && overlap_result.error() == uni::simd::Result::overlapping_buffers);

    auto aliased_input = make_input(17U);
    uni::simd::PfbChannelizerBlock input_overlap{
        .input = as_components(std::span<const Complex>{aliased_input}),
    };
    input_overlap.outputs[0] = as_components(std::span<Complex>{aliased_input}.first(5U));
    input_overlap.outputs[1] = as_components(std::span<Complex>{full_output});
    const auto input_overlap_result = channelizer->process(input_overlap);
    assert(!input_overlap_result && input_overlap_result.error() == uni::simd::Result::overlapping_buffers);

    std::array<float, 3U> odd_components{};
    uni::simd::PfbChannelizerBlock odd_block{.input = odd_components};
    odd_block.outputs[0] = as_components(std::span<Complex>{full_output});
    odd_block.outputs[1] = as_components(std::span<Complex>{short_output});
    const auto odd_result = channelizer->process(odd_block);
    assert(!odd_result && odd_result.error() == uni::simd::Result::invalid_size);
}

void test_scalar_reference(const uni::simd::Context& generic) {
    constexpr std::array<std::size_t, 14U> tap_counts{1U, 2U, 3U, 7U, 8U, 9U, 31U, 32U, 33U, 168U, 169U, 257U, 1024U, 1025U};
    const auto input = make_input(43U);
    for (const std::size_t bin_count : {4U, 8U, 16U, 32U}) {
        for (std::size_t decimation = 1U; decimation <= bin_count; decimation *= 2U) {
            for (const auto grid : {uni::simd::PfbGridOffset::integer_bins, uni::simd::PfbGridOffset::half_bins}) {
                for (const std::size_t tap_count : tap_counts) {
                    const auto taps = make_taps(tap_count);
                    std::array<std::int32_t, uni::simd::pfb_channelizer_max_outputs> selected{};
                    const std::size_t selected_count = std::min<std::size_t>(selected.size(), bin_count);
                    for (std::size_t index = 0U; index < selected_count; ++index) {
                        selected[index] = static_cast<std::int32_t>(index) - static_cast<std::int32_t>(bin_count / 2U);
                    }
                    const uni::simd::PfbChannelizerConfig config{
                        .bin_count = bin_count, .decimation = decimation, .grid_offset = grid, .taps = taps,
                        .logical_bins = {selected.data(), selected_count}};
                    compare_reference(run(generic, config, input), direct_reference(config, input));
                }
            }
        }
    }
    constexpr std::array<std::int32_t, 1U> selected{-2};
    const auto taps = make_taps(33U);
    for (const auto grid : {uni::simd::PfbGridOffset::integer_bins,
                            uni::simd::PfbGridOffset::half_bins}) {
        const uni::simd::PfbChannelizerConfig config{
            .bin_count = 8U, .decimation = 4U, .grid_offset = grid,
            .taps = taps, .logical_bins = selected};
        compare_reference(run(generic, config, input), direct_reference(config, input));
    }
}

void test_streaming_alignment_and_wrap(const uni::simd::Context& generic) {
    constexpr std::array<std::int32_t, 8U> bins{-4, -3, -2, -1, 0, 1, 2, 3};
    constexpr std::array<std::size_t, 9U> splits{1U, 2U, 3U, 5U, 7U, 17U, 31U, 64U, 11U};
    const auto taps = make_taps(1025U);
    auto storage = make_input(5004U);
    const std::span<const Complex> unaligned_input{storage.data() + 1U, 5003U};
    const uni::simd::PfbChannelizerConfig config{
        .bin_count = 8U, .decimation = 4U, .grid_offset = uni::simd::PfbGridOffset::half_bins,
        .taps = taps, .logical_bins = bins};
    const auto whole = run(generic, config, unaligned_input);
    const auto split = run(generic, config, unaligned_input, splits);
    assert(whole.outputs == split.outputs);
    compare_reference(whole, direct_reference(config, unaligned_input), 5.0e-5f, 7.0e-4f);

    const uni::simd::PfbChannelizerConfig no_outputs{
        .bin_count = 8U, .decimation = 4U, .grid_offset = uni::simd::PfbGridOffset::half_bins, .taps = taps};
    const auto zero = run(generic, no_outputs, unaligned_input, splits);
    assert(zero.outputs.empty() && zero.produced == 1U + (unaligned_input.size() - 1U) / 4U);
}

void test_dispatch(const uni::simd::Context& generic) {
    constexpr std::array<std::int32_t, 4U> bins{-2, -1, 0, 1};
    const auto taps = make_taps(169U);
    const auto input = make_input(1031U);
    const uni::simd::PfbChannelizerConfig accelerated{
        .bin_count = 8U, .decimation = 4U, .grid_offset = uni::simd::PfbGridOffset::half_bins,
        .taps = taps, .logical_bins = bins};
    const auto reference = run(generic, accelerated, input);
    assert(reference.backend == uni::simd::Backend::generic);
    constexpr std::array<std::size_t, 9U> splits{1U, 2U, 3U, 4U, 5U, 7U, 17U, 31U, 11U};

    for (const auto backend : {uni::simd::Backend::avx2_fma, uni::simd::Backend::avx512,
                               uni::simd::Backend::neon}) {
        const auto context = uni::simd::create_context({.backend = backend});
        if (!context) {
            assert(context.error() == uni::simd::Result::unsupported_backend);
            continue;
        }
        const auto actual = run(*context, accelerated, input);
        if (backend == uni::simd::Backend::avx512) {
            assert(actual.backend == uni::simd::Backend::avx2_fma ||
                   actual.backend == uni::simd::Backend::generic);
        } else {
            assert(actual.backend == backend);
        }
        const bool avx512_pfb_available = backend != uni::simd::Backend::avx512 ||
                                          actual.backend == uni::simd::Backend::avx2_fma;
        const auto resolved_backend_for = [&](const std::size_t bin_count) {
            if (backend != uni::simd::Backend::avx512) {
                return backend;
            }
            if (!avx512_pfb_available) {
                return uni::simd::Backend::generic;
            }
            return bin_count == 32U ? uni::simd::Backend::avx512
                                    : uni::simd::Backend::avx2_fma;
        };
        compare_runs(reference, actual);
        const auto split = run(*context, accelerated, input, splits);
        compare_runs(actual, split);

        const auto compare_accelerated = [&](const uni::simd::PfbChannelizerConfig& config) {
            const auto generic_result = run(generic, config, input);
            const auto accelerated_result = run(*context, config, input);
            assert(accelerated_result.backend == resolved_backend_for(config.bin_count));
            compare_runs(generic_result, accelerated_result);
            const auto fragmented = run(*context, config, input, splits);
            compare_runs(accelerated_result, fragmented);
        };

        for (const std::size_t bin_count : {4U, 8U, 16U, 32U}) {
            const auto matrix_taps = make_taps(bin_count + 1U);
            std::array<std::int32_t, 3U> selected{
                -static_cast<std::int32_t>(bin_count / 2U), 0,
                static_cast<std::int32_t>(bin_count / 2U) - 1};
            for (std::size_t decimation = 1U; decimation <= bin_count; decimation *= 2U) {
                for (const auto grid : {uni::simd::PfbGridOffset::integer_bins,
                                        uni::simd::PfbGridOffset::half_bins}) {
                    compare_accelerated({.bin_count = bin_count, .decimation = decimation,
                                         .grid_offset = grid, .taps = matrix_taps,
                                         .logical_bins = selected});
                }
            }

            for (const std::size_t tap_count :
                 std::array<std::size_t, 4U>{1U, bin_count - 1U, bin_count, bin_count + 1U}) {
                const auto boundary_taps = make_taps(tap_count);
                compare_accelerated({.bin_count = bin_count, .decimation = bin_count / 2U,
                                     .grid_offset = uni::simd::PfbGridOffset::integer_bins,
                                     .taps = boundary_taps, .logical_bins = selected});
            }

            const auto batched_taps = make_taps(4U * bin_count + 1U);
            compare_accelerated({.bin_count = bin_count, .decimation = bin_count / 2U,
                                 .grid_offset = uni::simd::PfbGridOffset::half_bins,
                                 .taps = batched_taps, .logical_bins = selected});

            const std::array one_short_bin{static_cast<std::int32_t>(bin_count / 2U) - 1};
            compare_accelerated({.bin_count = bin_count, .decimation = bin_count,
                                 .grid_offset = uni::simd::PfbGridOffset::integer_bins,
                                 .taps = matrix_taps, .logical_bins = one_short_bin});

            const auto maximum_taps = make_taps(uni::simd::pfb_channelizer_max_taps);
            const std::array one_bin{static_cast<std::int32_t>(bin_count / 2U) - 1};
            compare_accelerated({.bin_count = bin_count, .decimation = bin_count / 2U,
                                 .grid_offset = uni::simd::PfbGridOffset::half_bins,
                                 .taps = maximum_taps, .logical_bins = one_bin});

            const uni::simd::PfbChannelizerConfig no_outputs{
                .bin_count = bin_count, .decimation = bin_count,
                .grid_offset = uni::simd::PfbGridOffset::integer_bins,
                .taps = matrix_taps};
            const auto discarded = run(*context, no_outputs, input, splits);
            assert(discarded.backend == resolved_backend_for(bin_count) && discarded.outputs.empty());
        }

        const auto long_input = make_input(5003U);
        const auto long_taps = make_taps(uni::simd::pfb_channelizer_max_taps);
        constexpr std::array<std::int32_t, 1U> long_bin{15};
        const uni::simd::PfbChannelizerConfig long_config{
            .bin_count = 32U, .decimation = 16U,
            .grid_offset = uni::simd::PfbGridOffset::half_bins,
            .taps = long_taps, .logical_bins = long_bin};
        const auto long_reference = run(generic, long_config, long_input);
        const auto long_accelerated = run(*context, long_config, long_input);
        assert(long_accelerated.backend == resolved_backend_for(32U));
        compare_runs(long_reference, long_accelerated);
        compare_runs(long_accelerated, run(*context, long_config, long_input, splits));

        for (const std::size_t tap_count : {168U, 170U}) {
            const auto general_taps = make_taps(tap_count);
            compare_accelerated({.bin_count = 8U, .decimation = 4U,
                                 .grid_offset = uni::simd::PfbGridOffset::half_bins,
                                 .taps = general_taps, .logical_bins = bins});
        }
        constexpr std::array<std::int32_t, 8U> all_bins{-4, -3, -2, -1, 0, 1, 2, 3};
        const auto maximum_eight_taps = make_taps(uni::simd::pfb_channelizer_max_taps);
        compare_accelerated({.bin_count = 8U, .decimation = 4U,
                             .grid_offset = uni::simd::PfbGridOffset::integer_bins,
                             .taps = maximum_eight_taps, .logical_bins = all_bins});
        compare_accelerated({.bin_count = 8U, .decimation = 4U,
                             .grid_offset = uni::simd::PfbGridOffset::half_bins,
                             .taps = maximum_eight_taps, .logical_bins = all_bins});

        constexpr std::array<std::int32_t, 8U> permuted_bins{3, -4, 1, -2, 0, 2, -1, -3};
        const auto count_taps = make_taps(170U);
        for (std::size_t selected_count = 2U; selected_count <= permuted_bins.size(); ++selected_count) {
            for (const auto grid : {uni::simd::PfbGridOffset::integer_bins,
                                    uni::simd::PfbGridOffset::half_bins}) {
                compare_accelerated({.bin_count = 8U, .decimation = 4U,
                                     .grid_offset = grid, .taps = count_taps,
                                     .logical_bins = {permuted_bins.data(), selected_count}});
            }
        }

        auto reset_channelizer = context->make_pfb_channelizer(accelerated);
        assert(reset_channelizer.has_value());
        const std::size_t count = *reset_channelizer->output_count(input.size());
        std::array<std::vector<Complex>, 4U> first_outputs;
        std::array<std::vector<Complex>, 4U> second_outputs;
        uni::simd::PfbChannelizerBlock first_block{
            .input = as_components(std::span<const Complex>{input}),
        };
        uni::simd::PfbChannelizerBlock second_block{
            .input = as_components(std::span<const Complex>{input}),
        };
        for (std::size_t output = 0U; output < first_outputs.size(); ++output) {
            first_outputs[output].resize(count);
            second_outputs[output].resize(count);
            first_block.outputs[output] = as_components(std::span<Complex>{first_outputs[output]});
            second_block.outputs[output] = as_components(std::span<Complex>{second_outputs[output]});
        }
        assert(reset_channelizer->process(first_block) == count);
        assert(reset_channelizer->reset() == uni::simd::Result::success);
        assert(reset_channelizer->process(second_block) == count);
        assert(first_outputs == second_outputs);

    }

    const auto deterministic = uni::simd::create_context({.math_mode = uni::simd::MathMode::deterministic});
    assert(deterministic.has_value());
    const auto first = run(*deterministic, accelerated, input);
    const auto second = run(*deterministic, accelerated, input);
    assert(first.backend == uni::simd::Backend::generic && first.outputs == second.outputs);
}

} // namespace

int main() {
    const auto generic = uni::simd::create_context({.backend = uni::simd::Backend::generic});
    assert(generic.has_value());
    test_validation(*generic);
    test_scalar_reference(*generic);
    test_streaming_alignment_and_wrap(*generic);
    test_dispatch(*generic);
    return 0;
}
