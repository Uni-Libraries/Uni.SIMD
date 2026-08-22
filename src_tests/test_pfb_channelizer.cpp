#ifdef NDEBUG
#undef NDEBUG
#endif

#include <uni/simd/simd.hpp>

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

struct RunResult {
    std::vector<std::vector<Complex>> outputs;
    std::size_t produced = 0U;
};

[[nodiscard]] std::vector<float> make_taps(const std::size_t count) {
    std::vector<float> taps(count);
    const float scale = 0.5f / std::sqrt(static_cast<float>(count));
    for (std::size_t index = 0U; index < count; ++index) {
        const float centered = static_cast<float>(static_cast<std::int64_t>(index) -
                                                  static_cast<std::int64_t>(count / 2U));
        taps[index] = scale * (0.7f * std::cos(centered * 0.173f) + 0.3f * std::sin(centered * 0.071f));
    }
    return taps;
}

[[nodiscard]] std::vector<Complex> make_input(const std::size_t count) {
    std::vector<Complex> input(count);
    std::uint32_t state = 0x12345678U;
    for (std::size_t index = 0U; index < count; ++index) {
        state = state * 1664525U + 1013904223U;
        const float real = static_cast<float>(static_cast<std::int32_t>(state >> 8U)) / 8388608.0f;
        state = state * 1664525U + 1013904223U;
        const float imag = static_cast<float>(static_cast<std::int32_t>(state >> 8U)) / 8388608.0f;
        input[index] = {real, imag};
    }
    return input;
}

[[nodiscard]] uni::simd::PfbChannelizerPlan make_plan(const std::size_t bins, const std::size_t decimation,
                                                       const uni::simd::PfbGridOffset offset,
                                                       const std::span<const float> taps,
                                                       const std::span<const std::int32_t> logical_bins) {
    uni::simd::PfbChannelizerPlan plan;
    const uni::simd::PfbChannelizerPlanConfig config{
        .bin_count = bins,
        .decimation = decimation,
        .grid_offset = offset,
        .taps = taps,
        .logical_bins = logical_bins,
    };
    assert(uni::simd::make_pfb_channelizer_plan(plan, config) == uni::simd::Result::success);
    return plan;
}

[[nodiscard]] RunResult run(const uni::simd::Context& context, const uni::simd::PfbChannelizerPlan& plan,
                            const std::span<const Complex> input, const std::span<const std::size_t> split_pattern = {},
                            const bool unchecked = false) {
    uni::simd::PfbChannelizerState state;
    assert(uni::simd::reset_pfb_channelizer_state(state, plan) == uni::simd::Result::success);
    const auto total_count = uni::simd::pfb_channelizer_output_count(plan, state, input.size());
    assert(total_count.has_value());

    RunResult result;
    result.outputs.resize(plan.selected_output_count());
    for (auto& output : result.outputs) {
        output.resize(*total_count);
    }

    std::size_t input_offset = 0U;
    std::size_t split_index = 0U;
    bool called = false;
    do {
        const std::size_t requested = split_pattern.empty() ? input.size() : split_pattern[split_index++ % split_pattern.size()];
        const std::size_t count = std::min(requested, input.size() - input_offset);
        uni::simd::PfbChannelizerBlockView block{.input = input.subspan(input_offset, count)};
        for (std::size_t output = 0U; output < result.outputs.size(); ++output) {
            block.outputs[output] = std::span<Complex>{result.outputs[output]}.subspan(result.produced);
        }

        std::size_t produced = 0U;
        if (unchecked) {
            produced = context.pfb_channelize_cf32_unchecked(plan, state, block);
        } else {
            assert(context.pfb_channelize_cf32(plan, state, block, produced) == uni::simd::Result::success);
        }
        result.produced += produced;
        input_offset += count;
        called = true;
    } while (input_offset < input.size() || !called);

    assert(result.produced == *total_count);
    return result;
}

[[nodiscard]] std::vector<std::vector<std::complex<double>>>
direct_reference(const std::size_t bins, const std::size_t decimation, const uni::simd::PfbGridOffset offset,
                 const std::span<const float> taps, const std::span<const std::int32_t> logical_bins,
                 const std::span<const Complex> input) {
    const std::size_t output_count = input.empty() ? 0U : 1U + (input.size() - 1U) / decimation;
    std::vector<std::vector<std::complex<double>>> outputs(logical_bins.size(),
                                                           std::vector<std::complex<double>>(output_count));
    const double delta = offset == uni::simd::PfbGridOffset::half_bins ? 0.5 : 0.0;
    for (std::size_t output = 0U; output < logical_bins.size(); ++output) {
        const double frequency_bin = static_cast<double>(logical_bins[output]) + delta;
        for (std::size_t hop = 0U; hop < output_count; ++hop) {
            const std::size_t sample_index = hop * decimation;
            std::complex<double> accumulator{};
            const std::size_t available_taps = std::min(taps.size(), sample_index + 1U);
            for (std::size_t tap = 0U; tap < available_taps; ++tap) {
                const std::size_t source_index = sample_index - tap;
                const double angle = -2.0 * pi * frequency_bin * static_cast<double>(source_index) /
                                     static_cast<double>(bins);
                const std::complex<double> source{input[source_index].real(), input[source_index].imag()};
                accumulator += static_cast<double>(taps[tap]) * source *
                               std::complex<double>{std::cos(angle), std::sin(angle)};
            }
            outputs[output][hop] = accumulator;
        }
    }
    return outputs;
}

void compare_reference(const RunResult& actual,
                       const std::vector<std::vector<std::complex<double>>>& expected,
                       const float relative_tolerance = 3.0e-4f) {
    assert(actual.outputs.size() == expected.size());
    for (std::size_t output = 0U; output < expected.size(); ++output) {
        assert(actual.outputs[output].size() == expected[output].size());
        for (std::size_t index = 0U; index < expected[output].size(); ++index) {
            const std::complex<float> reference{static_cast<float>(expected[output][index].real()),
                                                static_cast<float>(expected[output][index].imag())};
            const float tolerance = relative_tolerance * std::max(1.0f, std::abs(reference));
            assert(std::isfinite(actual.outputs[output][index].real()));
            assert(std::isfinite(actual.outputs[output][index].imag()));
            assert(std::abs(actual.outputs[output][index] - reference) <= tolerance);
        }
    }
}

void compare_backends(const RunResult& expected, const RunResult& actual, const float relative_tolerance) {
    assert(actual.outputs.size() == expected.outputs.size());
    assert(actual.produced == expected.produced);
    for (std::size_t output = 0U; output < expected.outputs.size(); ++output) {
        for (std::size_t index = 0U; index < expected.outputs[output].size(); ++index) {
            const float tolerance = relative_tolerance * std::max(1.0f, std::abs(expected.outputs[output][index]));
            assert(std::abs(expected.outputs[output][index] - actual.outputs[output][index]) <= tolerance);
        }
    }
}

void test_validation_and_transactionality(const uni::simd::Context& generic) {
    using uni::simd::PfbGridOffset;
    using uni::simd::Result;

    const auto taps = make_taps(17U);
    constexpr std::array bins{std::int32_t{-1}, std::int32_t{1}};
    auto plan = make_plan(8U, 4U, PfbGridOffset::half_bins, taps, bins);
    const auto original_bins = plan.logical_bins();
    assert(plan.bin_count() == 8U && plan.decimation() == 4U && plan.tap_count() == 17U);

    const std::array<float, 1U> one_tap{1.0f};
    const uni::simd::PfbChannelizerPlanConfig bad_m{.bin_count = 3U, .decimation = 1U, .taps = one_tap};
    assert(uni::simd::make_pfb_channelizer_plan(plan, bad_m) == Result::invalid_argument);
    assert(plan.bin_count() == 8U && plan.logical_bins().size() == original_bins.size());

    const uni::simd::PfbChannelizerPlanConfig bad_d{.bin_count = 8U, .decimation = 3U, .taps = one_tap};
    assert(uni::simd::make_pfb_channelizer_plan(plan, bad_d) == Result::invalid_argument);
    std::vector<float> too_many_taps(uni::simd::pfb_channelizer_max_taps + 1U, 1.0f);
    const uni::simd::PfbChannelizerPlanConfig bad_length{.bin_count = 8U, .decimation = 4U, .taps = too_many_taps};
    assert(uni::simd::make_pfb_channelizer_plan(plan, bad_length) == Result::invalid_argument);
    std::array<float, 1U> nonfinite{std::numeric_limits<float>::infinity()};
    const uni::simd::PfbChannelizerPlanConfig bad_tap{.bin_count = 8U, .decimation = 4U, .taps = nonfinite};
    assert(uni::simd::make_pfb_channelizer_plan(plan, bad_tap) == Result::invalid_argument);
    constexpr std::array duplicate_bins{std::int32_t{0}, std::int32_t{0}};
    const uni::simd::PfbChannelizerPlanConfig duplicate{.bin_count = 8U, .decimation = 4U, .taps = one_tap,
                                                        .logical_bins = duplicate_bins};
    assert(uni::simd::make_pfb_channelizer_plan(plan, duplicate) == Result::invalid_argument);
    constexpr std::array out_of_range{std::int32_t{4}};
    const uni::simd::PfbChannelizerPlanConfig bad_bin{.bin_count = 8U, .decimation = 4U, .taps = one_tap,
                                                      .logical_bins = out_of_range};
    assert(uni::simd::make_pfb_channelizer_plan(plan, bad_bin) == Result::invalid_argument);

    const auto input = make_input(19U);
    uni::simd::PfbChannelizerState failed_state;
    uni::simd::PfbChannelizerState control_state;
    assert(uni::simd::reset_pfb_channelizer_state(failed_state, plan) == Result::success);
    assert(uni::simd::reset_pfb_channelizer_state(control_state, plan) == Result::success);
    const auto expected_count = uni::simd::pfb_channelizer_output_count(plan, failed_state, input.size());
    assert(expected_count.has_value() && *expected_count == 5U);

    std::array<Complex, 4U> short_output{};
    std::array<Complex, 5U> second_output{};
    uni::simd::PfbChannelizerBlockView short_block{.input = input};
    short_block.outputs[0] = short_output;
    short_block.outputs[1] = second_output;
    std::size_t produced = 777U;
    assert(generic.pfb_channelize_cf32(plan, failed_state, short_block, produced) == Result::invalid_size);
    assert(produced == 777U);

    std::array<Complex, 5U> failed_output0{};
    std::array<Complex, 5U> failed_output1{};
    std::array<Complex, 5U> control_output0{};
    std::array<Complex, 5U> control_output1{};
    uni::simd::PfbChannelizerBlockView failed_block{.input = input};
    failed_block.outputs[0] = failed_output0;
    failed_block.outputs[1] = failed_output1;
    uni::simd::PfbChannelizerBlockView control_block{.input = input};
    control_block.outputs[0] = control_output0;
    control_block.outputs[1] = control_output1;
    assert(generic.pfb_channelize_cf32(plan, failed_state, failed_block, produced) == Result::success);
    std::size_t control_produced = 0U;
    assert(generic.pfb_channelize_cf32(plan, control_state, control_block, control_produced) == Result::success);
    assert(produced == control_produced && failed_output0 == control_output0 && failed_output1 == control_output1);

    auto overlap_input = make_input(8U);
    uni::simd::PfbChannelizerState overlap_state;
    assert(uni::simd::reset_pfb_channelizer_state(overlap_state, plan) == Result::success);
    uni::simd::PfbChannelizerBlockView overlap_block{.input = overlap_input};
    overlap_block.outputs[0] = std::span<Complex>{overlap_input}.first(2U);
    overlap_block.outputs[1] = second_output;
    assert(generic.pfb_channelize_cf32(plan, overlap_state, overlap_block, produced) == Result::overlapping_buffers);

    uni::simd::PfbChannelizerPlan uninitialized_plan;
    uni::simd::PfbChannelizerState uninitialized_state;
    assert(uni::simd::reset_pfb_channelizer_state(uninitialized_state, uninitialized_plan) == Result::invalid_argument);
    assert(!uni::simd::pfb_channelizer_output_count(uninitialized_plan, uninitialized_state, 1U).has_value());
}

void test_direct_equation_exhaustive(const uni::simd::Context& generic) {
    using uni::simd::PfbGridOffset;
    constexpr std::array<std::size_t, 23U> tap_lengths{
        1U, 2U, 3U, 4U, 5U, 7U, 8U, 9U, 15U, 16U, 17U, 31U,
        32U, 33U, 63U, 64U, 65U, 168U, 169U, 257U, 511U, 1024U, 1025U,
    };
    const auto input = make_input(41U);

    for (const std::size_t bins : {4U, 8U, 16U, 32U}) {
        for (std::size_t decimation = 1U; decimation <= bins; decimation *= 2U) {
            for (const PfbGridOffset offset : {PfbGridOffset::integer_bins, PfbGridOffset::half_bins}) {
                for (const std::size_t tap_count : tap_lengths) {
                    const auto taps = make_taps(tap_count);
                    for (std::int32_t first_bin = -static_cast<std::int32_t>(bins / 2U);
                         first_bin < static_cast<std::int32_t>(bins / 2U); first_bin += 8) {
                        std::array<std::int32_t, uni::simd::pfb_channelizer_max_outputs> selected{};
                        const std::size_t selected_count = std::min<std::size_t>(
                            8U, static_cast<std::size_t>(static_cast<std::int32_t>(bins / 2U) - first_bin));
                        for (std::size_t index = 0U; index < selected_count; ++index) {
                            selected[index] = first_bin + static_cast<std::int32_t>(index);
                        }
                        const std::span selected_span{selected.data(), selected_count};
                        const auto plan = make_plan(bins, decimation, offset, taps, selected_span);
                        const auto actual = run(generic, plan, input);
                        compare_reference(actual, direct_reference(bins, decimation, offset, taps, selected_span, input));
                    }
                }
            }
        }
    }
}

void test_first_hop_gain_and_impulse(const uni::simd::Context& generic) {
    using uni::simd::PfbGridOffset;
    constexpr std::array all_bins{std::int32_t{-4}, std::int32_t{-3}, std::int32_t{-2}, std::int32_t{-1},
                                  std::int32_t{0}, std::int32_t{1}, std::int32_t{2}, std::int32_t{3}};

    const std::array<float, 1U> unit_tap{1.0f};
    const auto gain_plan = make_plan(8U, 4U, PfbGridOffset::half_bins, unit_tap, all_bins);
    std::array<Complex, 1U> unit_impulse{Complex{1.0f, 0.0f}};
    const auto gain = run(generic, gain_plan, unit_impulse);
    assert(gain.produced == 1U);
    for (const auto& output : gain.outputs) {
        assert(std::abs(output[0] - Complex{1.0f, 0.0f}) < 2.0e-6f);
    }

    std::array<float, 169U> delayed_taps{};
    delayed_taps[84U] = 1.0f;
    const auto impulse_plan = make_plan(8U, 4U, PfbGridOffset::half_bins, delayed_taps, all_bins);
    std::array<Complex, 89U> impulse{};
    impulse[0] = {1.0f, 0.0f};
    const auto response = run(generic, impulse_plan, impulse);
    assert(response.produced == 23U);
    for (const auto& output : response.outputs) {
        assert(std::abs(output[21U] - Complex{1.0f, 0.0f}) < 3.0e-6f);
        for (std::size_t index = 0U; index < output.size(); ++index) {
            if (index != 21U) {
                assert(output[index] == Complex{});
            }
        }
    }

    const auto residue_taps = make_taps(169U);
    const auto residue_plan = make_plan(8U, 4U, PfbGridOffset::half_bins, residue_taps, all_bins);
    for (std::size_t residue = 0U; residue < 8U; ++residue) {
        std::vector<Complex> residue_impulse(211U);
        residue_impulse[residue] = {1.0f, -0.25f};
        compare_reference(run(generic, residue_plan, residue_impulse),
                          direct_reference(8U, 4U, PfbGridOffset::half_bins, residue_taps, all_bins,
                                           residue_impulse));
    }
}

void test_splits_wrap_canaries_and_zero_outputs(const uni::simd::Context& generic) {
    using uni::simd::PfbGridOffset;
    constexpr std::array all_bins{std::int32_t{-4}, std::int32_t{-3}, std::int32_t{-2}, std::int32_t{-1},
                                  std::int32_t{0}, std::int32_t{1}, std::int32_t{2}, std::int32_t{3}};
    constexpr std::array<std::size_t, 10U> split_pattern{1U, 2U, 3U, 4U, 5U, 7U, 17U, 31U, 6U, 19U};
    const auto taps = make_taps(1025U);
    const auto input = make_input(5003U);
    const auto plan = make_plan(8U, 4U, PfbGridOffset::half_bins, taps, all_bins);
    const auto whole = run(generic, plan, input);
    const auto split = run(generic, plan, input, split_pattern);
    assert(whole.produced == split.produced && whole.outputs == split.outputs);
    std::array<std::size_t, 64U> random_splits{};
    std::uint32_t random = 0x31415926U;
    for (std::size_t& size : random_splits) {
        random = random * 1664525U + 1013904223U;
        size = 1U + random % 97U;
    }
    const auto randomly_split = run(generic, plan, input, random_splits);
    assert(whole.produced == randomly_split.produced && whole.outputs == randomly_split.outputs);

    constexpr std::array max_tap_reference_bin{std::int32_t{3}};
    const auto max_tap_reference_plan =
        make_plan(8U, 4U, PfbGridOffset::half_bins, taps, max_tap_reference_bin);
    compare_reference(run(generic, max_tap_reference_plan, input),
                      direct_reference(8U, 4U, PfbGridOffset::half_bins, taps,
                                       max_tap_reference_bin, input),
                      6.0e-4f);

    uni::simd::PfbChannelizerState state;
    assert(uni::simd::reset_pfb_channelizer_state(state, plan) == uni::simd::Result::success);
    const std::size_t expected_count = 1U + (input.size() - 1U) / 4U;
    constexpr Complex canary{12345.0f, -6789.0f};
    std::vector<std::vector<Complex>> guarded(8U, std::vector<Complex>(expected_count + 2U, canary));
    uni::simd::PfbChannelizerBlockView block{.input = input};
    for (std::size_t output = 0U; output < guarded.size(); ++output) {
        block.outputs[output] = std::span<Complex>{guarded[output]}.subspan(1U, expected_count);
    }
    std::size_t produced = 0U;
    assert(generic.pfb_channelize_cf32(plan, state, block, produced) == uni::simd::Result::success);
    assert(produced == expected_count);
    for (const auto& output : guarded) {
        assert(output.front() == canary && output.back() == canary);
    }

    const std::span<const std::int32_t> no_bins;
    const auto zero_plan = make_plan(8U, 4U, PfbGridOffset::half_bins, taps, no_bins);
    const auto zero = run(generic, zero_plan, input, split_pattern, true);
    assert(zero.outputs.empty() && zero.produced == expected_count);
}

void test_backend_dispatch(const uni::simd::Context& generic) {
    using uni::simd::Backend;
    using uni::simd::Kernel;
    using uni::simd::PfbGridOffset;
    using uni::simd::Result;

    constexpr std::array all_bins{std::int32_t{-4}, std::int32_t{-3}, std::int32_t{-2}, std::int32_t{-1},
                                  std::int32_t{0}, std::int32_t{1}, std::int32_t{2}, std::int32_t{3}};
    constexpr std::array target_four_bins{std::int32_t{-2}, std::int32_t{-1},
                                          std::int32_t{0}, std::int32_t{1}};
    const auto taps = make_taps(169U);
    const auto input = make_input(1031U);
    const auto target_plan = make_plan(8U, 4U, PfbGridOffset::half_bins, taps, all_bins);
    const auto reference = run(generic, target_plan, input);
    const auto target_four_plan = make_plan(8U, 4U, PfbGridOffset::half_bins, taps, target_four_bins);
    const auto target_four_reference = run(generic, target_four_plan, input);
    compare_reference(target_four_reference,
                      direct_reference(8U, 4U, PfbGridOffset::half_bins, taps, target_four_bins, input));

    constexpr std::array<std::size_t, 9U> backend_splits{1U, 2U, 3U, 4U, 5U, 7U, 17U, 31U, 11U};
    for (const Backend backend : {Backend::avx2_fma, Backend::neon}) {
        const auto context = uni::simd::create_context({.backend = backend});
        if (!context.has_value()) {
            assert(context.error() == Result::unsupported_backend);
            continue;
        }
        const Backend selected_backend = context->kernel_backend(Kernel::pfb_channelizer_cf32);
        assert(selected_backend == backend || backend == Backend::avx512);
        const auto backend_whole = run(*context, target_plan, input, {}, true);
        compare_backends(reference, backend_whole, 5.0e-4f);
        const auto backend_split = run(*context, target_plan, input, backend_splits, true);
        assert(backend_whole.outputs == backend_split.outputs);
        const auto target_four_whole = run(*context, target_four_plan, input, {}, true);
        compare_backends(target_four_reference, target_four_whole, 5.0e-4f);
        const auto target_four_split = run(*context, target_four_plan, input, backend_splits, true);
        assert(target_four_whole.outputs == target_four_split.outputs);

        constexpr std::array fallback_bins{std::int32_t{-2}, std::int32_t{3}};
        const auto fallback_plan = make_plan(16U, 8U, PfbGridOffset::integer_bins, taps, fallback_bins);
        const auto fallback_reference = run(generic, fallback_plan, input);
        const auto fallback = run(*context, fallback_plan, input, {}, true);
        assert(fallback.outputs == fallback_reference.outputs);
    }

    const auto deterministic = uni::simd::create_context({.backend = Backend::automatic,
                                                           .math_mode = uni::simd::MathMode::deterministic});
    assert(deterministic.has_value());
    assert(deterministic->kernel_backend(Kernel::pfb_channelizer_cf32) == Backend::generic);
    const auto repeat0 = run(*deterministic, target_plan, input);
    const auto repeat1 = run(*deterministic, target_plan, input);
    assert(repeat0.outputs == repeat1.outputs);
}

} // namespace

int main() {
    const auto generic_result = uni::simd::create_context({.backend = uni::simd::Backend::generic});
    assert(generic_result.has_value());
    const auto& generic = *generic_result;

    test_validation_and_transactionality(generic);
    test_direct_equation_exhaustive(generic);
    test_first_hop_gain_and_impulse(generic);
    test_splits_wrap_canaries_and_zero_outputs(generic);
    test_backend_dispatch(generic);
    return 0;
}
