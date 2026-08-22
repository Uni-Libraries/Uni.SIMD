#include <uni/simd/simd.hpp>

#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "cpu_topology.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using Complex = std::complex<float>;
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr std::size_t block_samples = 131072U;
volatile float checksum_sink = 0.0f;

struct Config {
    std::size_t iterations = 20U;
    std::size_t warmup = 3U;
};

[[nodiscard]] bool parse_positive(const std::string_view text, std::size_t& value) {
    std::uint64_t parsed = 0U;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (error != std::errc{} || end != text.data() + text.size() || parsed == 0U) {
        return false;
    }
    value = static_cast<std::size_t>(parsed);
    return static_cast<std::uint64_t>(value) == parsed;
}

[[nodiscard]] bool parse_arguments(const int argc, char** argv, Config& config) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: " << argv[0] << " [--iterations N] [--warmup N]\n";
            return false;
        }
        if ((argument != "--iterations" && argument != "--warmup") || ++index == argc) {
            std::cerr << "Invalid argument: " << argument << '\n';
            return false;
        }
        std::size_t value = 0U;
        if (!parse_positive(argv[index], value)) {
            std::cerr << "Invalid positive integer: " << argv[index] << '\n';
            return false;
        }
        if (argument == "--iterations") {
            config.iterations = value;
        } else {
            config.warmup = value;
        }
    }
    return true;
}

[[nodiscard]] std::array<float, 169U> make_target_rrc() {
    constexpr double sample_rate = 200000000.0;
    constexpr double symbol_rate = 14336000.0;
    constexpr double beta = 0.2;
    std::array<float, 169U> taps{};
    double energy = 0.0;
    for (std::size_t index = 0U; index < taps.size(); ++index) {
        const double time = (static_cast<double>(index) - 84.0) * symbol_rate / sample_rate;
        double value = 0.0;
        if (std::abs(time) < 1.0e-14) {
            value = 1.0 + beta * (4.0 / pi - 1.0);
        } else if (std::abs(std::abs(4.0 * beta * time) - 1.0) < 1.0e-12) {
            value = beta / std::sqrt(2.0) *
                    ((1.0 + 2.0 / pi) * std::sin(pi / (4.0 * beta)) +
                     (1.0 - 2.0 / pi) * std::cos(pi / (4.0 * beta)));
        } else {
            value = (std::sin(pi * time * (1.0 - beta)) +
                     4.0 * beta * time * std::cos(pi * time * (1.0 + beta))) /
                    (pi * time * (1.0 - 16.0 * beta * beta * time * time));
        }
        taps[index] = static_cast<float>(value);
        energy += value * value;
    }
    const float normalization = static_cast<float>(1.0 / std::sqrt(energy));
    for (float& tap : taps) {
        tap *= normalization;
    }
    return taps;
}

[[nodiscard]] std::vector<Complex> make_input() {
    std::vector<Complex> input(block_samples);
    std::uint32_t random = 0x9e3779b9U;
    for (std::size_t index = 0U; index < input.size(); ++index) {
        random = random * 1664525U + 1013904223U;
        const float real = static_cast<float>(static_cast<std::int32_t>(random >> 8U)) / 8388608.0f;
        random = random * 1664525U + 1013904223U;
        const float imag = static_cast<float>(static_cast<std::int32_t>(random >> 8U)) / 8388608.0f;
        input[index] = {real, imag};
    }
    return input;
}

[[nodiscard]] std::span<const std::int32_t> selected_bins(const std::size_t output_count,
                                                          const std::array<std::int32_t, 8U>& bins) {
    if (output_count == 0U) {
        return {};
    }
    if (output_count == 1U) {
        return {bins.data() + 4U, 1U};
    }
    if (output_count == 4U) {
        return {bins.data() + 2U, 4U};
    }
    return bins;
}

void benchmark_case(const Config& config, const uni::simd::Backend requested_backend,
                    const std::size_t output_count, const std::span<const Complex> input,
                    const std::span<const float> taps) {
    const auto context = uni::simd::create_context({.backend = requested_backend});
    if (!context.has_value()) {
        return;
    }

    constexpr std::array<std::int32_t, 8U> bins{-4, -3, -2, -1, 0, 1, 2, 3};
    uni::simd::PfbChannelizerPlan plan;
    const uni::simd::PfbChannelizerPlanConfig plan_config{
        .bin_count = 8U,
        .decimation = 4U,
        .grid_offset = uni::simd::PfbGridOffset::half_bins,
        .taps = taps,
        .logical_bins = selected_bins(output_count, bins),
    };
    if (uni::simd::make_pfb_channelizer_plan(plan, plan_config) != uni::simd::Result::success) {
        throw std::runtime_error("PFB plan construction failed");
    }

    uni::simd::PfbChannelizerState state;
    if (uni::simd::reset_pfb_channelizer_state(state, plan) != uni::simd::Result::success) {
        throw std::runtime_error("PFB state reset failed");
    }
    constexpr std::size_t output_samples = block_samples / 4U;
    std::array<std::vector<Complex>, 8U> outputs;
    uni::simd::PfbChannelizerBlockView block{.input = input};
    for (std::size_t output = 0U; output < output_count; ++output) {
        outputs[output].resize(output_samples);
        block.outputs[output] = outputs[output];
    }

    for (std::size_t iteration = 0U; iteration < config.warmup; ++iteration) {
        if (context->pfb_channelize_cf32_unchecked(plan, state, block) != output_samples) {
            throw std::runtime_error("PFB warmup produced an unexpected sample count");
        }
    }

    const auto begin = Clock::now();
    for (std::size_t iteration = 0U; iteration < config.iterations; ++iteration) {
        if (context->pfb_channelize_cf32_unchecked(plan, state, block) != output_samples) {
            throw std::runtime_error("PFB benchmark produced an unexpected sample count");
        }
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
    const double processed = static_cast<double>(block_samples) * static_cast<double>(config.iterations);
    const double msps = processed / seconds / 1.0e6;
    const double nanoseconds = seconds * 1.0e9 / processed;
    float checksum = 0.0f;
    for (std::size_t output = 0U; output < output_count; ++output) {
        checksum += outputs[output].front().real() + outputs[output].back().imag();
    }
    checksum_sink = checksum;

    const auto selected_backend = context->kernel_backend(uni::simd::Kernel::pfb_channelizer_cf32);
    std::cout << std::left << std::setw(11) << uni::simd::backend_name(requested_backend)
              << std::setw(11) << uni::simd::backend_name(selected_backend)
              << std::right << std::setw(8) << output_count
              << std::setw(14) << std::fixed << std::setprecision(2) << msps
              << std::setw(14) << std::setprecision(3) << nanoseconds << '\n';
}

} // namespace

int main(const int argc, char** argv) {
    Config config;
    if (!parse_arguments(argc, argv, config)) {
        return argc > 1 && (std::string_view{argv[1]} == "--help" || std::string_view{argv[1]} == "-h") ? 0 : 1;
    }

    const auto taps = make_target_rrc();
    const auto input = make_input();
    namespace topology = uni::simd::benchmark::cpu_topology;
    const auto topology_result = topology::query_snapshot();
    const auto core_classes = topology::build_core_classes(topology_result.snapshot);
    std::unique_ptr<topology::ScopedThreadAffinity> affinity;
    if (!core_classes.empty()) {
        affinity = std::make_unique<topology::ScopedThreadAffinity>(core_classes.front().logical_processor);
        if (!affinity->can_run()) {
            affinity.reset();
        }
    }
    std::cout << "PFB CF32 M=8 D=4 half-bin, 169-tap RRC, block=" << block_samples << "\n";
    if (affinity) {
        std::cout << "Pinned CPU " << core_classes.front().logical_processor.logical_processor_id << " ("
                  << core_classes.front().label << ")\n";
    }
    std::cout << std::left << std::setw(11) << "requested" << std::setw(11) << "selected"
              << std::right << std::setw(8) << "outputs" << std::setw(14) << "input MSPS"
              << std::setw(14) << "ns/input" << '\n';
    for (const auto backend : {uni::simd::Backend::generic, uni::simd::Backend::avx2_fma,
                               uni::simd::Backend::neon, uni::simd::Backend::automatic}) {
        for (const std::size_t output_count : {0U, 1U, 4U, 8U}) {
            benchmark_case(config, backend, output_count, input, taps);
        }
    }
    return 0;
}
