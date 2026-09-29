#include "uni_simd.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <random>
#include <vector>

namespace {

void Check(const bool condition, const char* const message) {
    if (!condition) {
        std::fprintf(stderr, "symbol_sync4 test failed: %s\n", message);
        std::abort();
    }
}

using Complex = std::complex<float>;
using Channel = std::vector<Complex>;
constexpr std::size_t kLanes = UNI_SIMD_SYMBOL_SYNC4_CHANNEL_COUNT;

struct LoopSettings {
    double omega{3.4877};
    double deviation{0.03};
    double loop_bw{0.003};
    double damping{1.0};
    bool auto_ted{true};
    uni_simd_symbol_sync_interpolator_e interpolator{UNI_SIMD_SYMBOL_SYNC_INTERPOLATOR_CUBIC};
    uni_simd_symbol_sync_ted_e ted{UNI_SIMD_SYMBOL_SYNC_TED_GARDNER};
    double mm_threshold{0.0005};
    double fallback_threshold{0.003};
};

[[nodiscard]] double Alpha(const LoopSettings& s) {
    return 4.0 * s.damping * s.loop_bw / (1.0 + 2.0 * s.damping * s.loop_bw + s.loop_bw * s.loop_bw);
}
[[nodiscard]] double Beta(const LoopSettings& s) { return 4.0 * s.loop_bw * s.loop_bw / (1.0 + 2.0 * s.damping * s.loop_bw + s.loop_bw * s.loop_bw); }

[[nodiscard]] uni_simd_symbol_sync4_config_t MakeConfig(const LoopSettings& s) {
    uni_simd_symbol_sync4_config_t config{};
    config.descriptor_size = UNI_SIMD_SYMBOL_SYNC4_CONFIG_DESCRIPTOR_SIZE;
    config.interpolator = s.interpolator;
    config.ted = s.ted;
    for (std::size_t lane = 0U; lane < kLanes; ++lane) {
        config.omega[lane] = s.omega;
        config.omega_min[lane] = s.omega * (1.0 - s.deviation);
        config.omega_max[lane] = s.omega * (1.0 + s.deviation);
        config.alpha[lane] = Alpha(s);
        config.beta[lane] = Beta(s);
    }
    config.auto_ted = s.auto_ted ? 1U : 0U;
    config.auto_ted_fallback = 1U;
    config.lock_ema_alpha = 0.001;
    config.mm_min_symbols = 4096U;
    config.mm_hold_symbols = 1024U;
    config.fallback_hold_symbols = 512U;
    config.mm_threshold = s.mm_threshold;
    config.fallback_threshold = s.fallback_threshold;
    return config;
}

[[nodiscard]] uni_simd_kernel_t* Create(const uni_simd_symbol_sync4_config_t& config, const uni_simd_backend_e backend) {
    auto* const kernel = uni_simd_kernel_create(UNI_SIMD_KERNEL_SYMBOL_SYNC4_CF32);
    Check(kernel != nullptr, "create");
    std::array params{
        uni_simd_param_t{UNI_SIMD_PARAM_CONFIG, {.const_pointer = &config}},
        uni_simd_param_t{UNI_SIMD_PARAM_BACKEND, {.u32 = backend}},
    };
    Check(uni_simd_kernel_param_set_many(kernel, params.data(), params.size()) == UNI_SIMD_RESULT_SUCCESS, "params");
    return kernel;
}

/** Raised-cosine shaped QPSK with a per-lane rate offset and delay (in samples). */
[[nodiscard]] Channel MakeSignal(const std::size_t samples, const double omega, const double delay, const float noise, const unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> gaussian(0.0f, noise);
    const auto symbol_count = static_cast<std::size_t>(static_cast<double>(samples) / omega) + 16U;
    std::vector<Complex> symbols(symbol_count);
    for (auto& symbol : symbols) {
        symbol = {(rng() & 1U) != 0U ? 0.7071f : -0.7071f, (rng() & 1U) != 0U ? 0.7071f : -0.7071f};
    }
    const auto pulse = [](const double t) {
        constexpr double rolloff = 0.35;
        if (std::abs(t) < 1e-9) return 1.0;
        if (std::abs(std::abs(2.0 * rolloff * t) - 1.0) < 1e-9) return std::numbers::pi / 4.0 * std::sin(std::numbers::pi * t) / (std::numbers::pi * t);
        return std::sin(std::numbers::pi * t) / (std::numbers::pi * t) * std::cos(std::numbers::pi * rolloff * t) / (1.0 - 4.0 * rolloff * rolloff * t * t);
    };
    Channel signal(samples);
    for (std::size_t n = 0U; n < samples; ++n) {
        const double position = (static_cast<double>(n) - delay) / omega;
        const auto centre = static_cast<std::ptrdiff_t>(std::floor(position));
        std::complex<double> sum{};
        for (std::ptrdiff_t k = centre - 8; k <= centre + 8; ++k) {
            if (k >= 0 && k < static_cast<std::ptrdiff_t>(symbol_count)) {
                const auto symbol = symbols[static_cast<std::size_t>(k)];
                sum += std::complex<double>(symbol.real(), symbol.imag()) * pulse(position - static_cast<double>(k));
            }
        }
        signal[n] = Complex(static_cast<float>(sum.real()) + gaussian(rng), static_cast<float>(sum.imag()) + gaussian(rng));
    }
    return signal;
}

/**
 * Independent scalar model of the loop in absolute time over the whole stream, written the
 * straightforward way (the original single-channel node), with zeros before the stream start.
 */
[[nodiscard]] Channel Oracle(const Channel& input, const uni_simd_symbol_sync4_config_t& config, const std::size_t lane) {
    const auto sample = [&](const std::ptrdiff_t index) {
        return index >= 0 && index < static_cast<std::ptrdiff_t>(input.size()) ? input[static_cast<std::size_t>(index)] : Complex{};
    };
    const auto interpolate = [&](const double t) {
        const auto n = static_cast<std::ptrdiff_t>(std::floor(t));
        const double mu = t - static_cast<double>(n);
        if (config.interpolator == UNI_SIMD_SYMBOL_SYNC_INTERPOLATOR_LINEAR) {
            return static_cast<float>(1.0 - mu) * sample(n) + static_cast<float>(mu) * sample(n + 1);
        }
        const double c0 = -mu * (mu - 1.0) * (mu - 2.0) / 6.0;
        const double c1 = (mu + 1.0) * (mu - 1.0) * (mu - 2.0) / 2.0;
        const double c2 = -(mu + 1.0) * mu * (mu - 2.0) / 2.0;
        const double c3 = (mu + 1.0) * mu * (mu - 1.0) / 6.0;
        return static_cast<float>(c0) * sample(n - 1) + static_cast<float>(c1) * sample(n) + static_cast<float>(c2) * sample(n + 1) +
               static_cast<float>(c3) * sample(n + 2);
    };
    const auto sign = [](const float value) { return value >= 0.0f ? 1.0f : -1.0f; };

    Channel output;
    double t = 1.0;
    double omega = config.omega[lane];
    bool mm = config.ted == UNI_SIMD_SYMBOL_SYNC_TED_MUELLER_MULLER;
    const bool auto_ted = config.auto_ted != 0U && !mm;
    Complex previous{};
    bool have_previous = false;
    double ema = 0.0;
    std::uint64_t lock = 0U;
    std::uint64_t unlock = 0U;
    while (std::floor(t) + 2.0 < static_cast<double>(input.size())) {
        const Complex current = interpolate(t);
        double step = omega;
        if (have_previous) {
            float error = 0.0f;
            float energy = std::norm(previous) + std::norm(current);
            if (mm) {
                error = (sign(previous.real()) * current.real() + sign(previous.imag()) * current.imag()) -
                        (sign(current.real()) * previous.real() + sign(current.imag()) * previous.imag());
            } else {
                const Complex mid = interpolate(t - 0.5 * omega);
                error = mid.real() * (previous.real() - current.real()) + mid.imag() * (previous.imag() - current.imag());
                energy += std::norm(mid);
            }
            omega = std::clamp(omega + config.beta[lane] * error, config.omega_min[lane], config.omega_max[lane]);
            step = omega + config.alpha[lane] * error;
            if (auto_ted) {
                const float normalised = error / (energy + 1e-12f);
                ema = (1.0 - config.lock_ema_alpha) * ema + config.lock_ema_alpha * static_cast<double>(normalised * normalised);
                const auto switch_ted = [&] {
                    mm = !mm;
                    ema = 0.0;
                    lock = 0U;
                    unlock = 0U;
                };
                if (!mm) {
                    if (output.size() >= config.mm_min_symbols) {
                        lock = ema < config.mm_threshold ? lock + 1U : 0U;
                        if (config.mm_hold_symbols > 0U && lock >= config.mm_hold_symbols) switch_ted();
                    }
                } else if (config.auto_ted_fallback != 0U) {
                    unlock = ema > config.fallback_threshold ? unlock + 1U : 0U;
                    if (config.fallback_hold_symbols > 0U && unlock >= config.fallback_hold_symbols) switch_ted();
                }
            }
        }
        have_previous = true;
        previous = current;
        output.push_back(current);
        t += step;
    }
    return output;
}

struct RunResult {
    std::array<Channel, kLanes> symbols;
    uni_simd_symbol_sync4_result_t last{};
};

/** Feeds the channels in blocks whose sizes come from `sizes` (cycled); lane k's block is offset by k samples. */
[[nodiscard]] RunResult Run(uni_simd_kernel_t* const kernel, const std::array<Channel, kLanes>& channels, const std::vector<std::size_t>& sizes) {
    RunResult run;
    std::array<std::size_t, kLanes> offset{};
    std::array<Channel, kLanes> output;
    std::size_t turn = 0U;
    for (;;) {
        uni_simd_symbol_sync4_block_t block{};
        block.descriptor_size = UNI_SIMD_SYMBOL_SYNC4_BLOCK_DESCRIPTOR_SIZE;
        bool any = false;
        for (std::size_t lane = 0U; lane < kLanes; ++lane) {
            const std::size_t size = sizes[(turn + lane) % sizes.size()];
            const std::size_t count = std::min(size, channels[lane].size() - offset[lane]);
            any = any || count != 0U;
            block.input[lane] = reinterpret_cast<const float*>(channels[lane].data() + offset[lane]);
            block.input_count[lane] = count;
            output[lane].assign(count / 2U + 8U, Complex{});
            block.output[lane] = reinterpret_cast<float*>(output[lane].data());
            block.output_capacity[lane] = output[lane].size();
        }
        if (!any) {
            break;
        }
        uni_simd_symbol_sync4_result_t result{};
        result.descriptor_size = UNI_SIMD_SYMBOL_SYNC4_RESULT_DESCRIPTOR_SIZE;
        Check(uni_simd_kernel_execute(kernel, &block, &result) == UNI_SIMD_RESULT_SUCCESS, "execute");
        Check(result.truncated_mask == 0U, "no truncation");
        for (std::size_t lane = 0U; lane < kLanes; ++lane) {
            run.symbols[lane].insert(run.symbols[lane].end(), output[lane].begin(),
                                     output[lane].begin() + static_cast<std::ptrdiff_t>(result.output_count[lane]));
            offset[lane] += block.input_count[lane];
        }
        run.last = result;
        ++turn;
    }
    return run;
}

[[nodiscard]] float MaxDifference(const Channel& left, const Channel& right) {
    float difference = 0.0f;
    for (std::size_t index = 0U; index < std::min(left.size(), right.size()); ++index) {
        difference = std::max(difference, std::abs(left[index] - right[index]));
    }
    return difference;
}

[[nodiscard]] std::array<Channel, kLanes> MakeChannels(const std::size_t samples, const double omega, const float noise) {
    std::array<Channel, kLanes> channels;
    for (std::size_t lane = 0U; lane < kLanes; ++lane) {
        // Different rate offsets and delays per lane, well inside the loop's deviation.
        channels[lane] = MakeSignal(samples, omega * (1.0 + 2e-4 * static_cast<double>(lane) - 3e-4), 0.37 * static_cast<double>(lane),
                                    noise, 11U + static_cast<unsigned>(lane));
    }
    return channels;
}

void TestMatchesOracle(const uni_simd_backend_e backend, const LoopSettings& settings, const char* const name, const std::uint64_t min_switches = 0U) {
    const auto channels = MakeChannels(60000U, settings.omega, 0.02f);
    const auto config = MakeConfig(settings);
    auto* const kernel = Create(config, backend);
    const auto run = Run(kernel, channels, {4096U, 1U, 777U, 0U, 3U, 12000U, 31U});
    for (std::size_t lane = 0U; lane < kLanes; ++lane) {
        const Channel expected = Oracle(channels[lane], config, lane);
        const Channel& actual = run.symbols[lane];
        // The kernel emits a symbol once all four stencil samples exist, as the oracle does.
        if (actual.size() != expected.size() || MaxDifference(actual, expected) > 2e-3f) {
            std::fprintf(stderr, "%s lane %zu: %zu vs %zu symbols, max difference %g\n", name, lane, actual.size(), expected.size(),
                         static_cast<double>(MaxDifference(actual, expected)));
            Check(false, "kernel matches the scalar model");
        }
    }
    for (std::size_t lane = 0U; lane < kLanes; ++lane) {
        Check(run.last.ted_switches[lane] >= min_switches, "detector switches as configured");
    }
    Check(uni_simd_kernel_free(kernel) == UNI_SIMD_RESULT_SUCCESS, "free");
}

void TestBackendsAgreeAndSplitInvariant() {
    const LoopSettings settings{};
    const auto channels = MakeChannels(80000U, settings.omega, 0.05f);
    const auto config = MakeConfig(settings);
    auto* const generic = Create(config, UNI_SIMD_BACKEND_GENERIC);
    auto* const automatic = Create(config, UNI_SIMD_BACKEND_AUTOMATIC);
    const auto reference = Run(generic, channels, {131072U});
    const auto split = Run(automatic, channels, {5U, 4000U, 2U, 9999U, 64U});
    for (std::size_t lane = 0U; lane < kLanes; ++lane) {
        Check(reference.symbols[lane].size() == split.symbols[lane].size(), "same symbol count across backends and splits");
        Check(MaxDifference(reference.symbols[lane], split.symbols[lane]) < 1e-3f, "same symbols across backends and splits");
        Check(reference.last.symbols[lane] == split.last.symbols[lane], "same symbol counter");
    }
    Check(uni_simd_kernel_reset(automatic) == UNI_SIMD_RESULT_SUCCESS, "reset");
    const auto again = Run(automatic, channels, {131072U});
    for (std::size_t lane = 0U; lane < kLanes; ++lane) {
        Check(MaxDifference(again.symbols[lane], split.symbols[lane]) < 1e-3f, "reset restores the initial state");
    }
    Check(uni_simd_kernel_free(generic) == UNI_SIMD_RESULT_SUCCESS, "free");
    Check(uni_simd_kernel_free(automatic) == UNI_SIMD_RESULT_SUCCESS, "free");
}

void TestAcquiresTimingAndSwitchesDetector() {
    // Gardner's own pattern noise on this signal keeps err_norm2 around 0.01.
    const LoopSettings settings{.mm_threshold = 0.02, .fallback_threshold = 0.2};
    const auto channels = MakeChannels(200000U, settings.omega, 0.01f);
    const auto config = MakeConfig(settings); // borrowed by the kernel until its first execution
    auto* const kernel = Create(config, UNI_SIMD_BACKEND_AUTOMATIC);
    const auto run = Run(kernel, channels, {16384U});
    for (std::size_t lane = 0U; lane < kLanes; ++lane) {
        const double true_omega = settings.omega * (1.0 + 2e-4 * static_cast<double>(lane) - 3e-4);
        Check(std::abs(run.last.omega[lane] - true_omega) < 2e-3, "omega converges to the symbol rate");
        Check(run.last.ted[lane] == UNI_SIMD_SYMBOL_SYNC_TED_MUELLER_MULLER && run.last.ted_switches[lane] == 1U, "switches to M&M once locked");
        // After acquisition every symbol lies close to a QPSK point.
        const Channel& symbols = run.symbols[lane];
        double error = 0.0;
        for (std::size_t index = symbols.size() / 2U; index < symbols.size(); ++index) {
            const Complex ideal{symbols[index].real() >= 0.0f ? 0.7071f : -0.7071f, symbols[index].imag() >= 0.0f ? 0.7071f : -0.7071f};
            error += std::norm(symbols[index] - ideal);
        }
        Check(error / static_cast<double>(symbols.size() - symbols.size() / 2U) < 0.01, "locked constellation");
    }
    Check(uni_simd_kernel_free(kernel) == UNI_SIMD_RESULT_SUCCESS, "free");
}

void TestValidation() {
    LoopSettings settings{};
    auto config = MakeConfig(settings);
    config.omega_min[2] = 0.5;
    auto* kernel = Create(config, UNI_SIMD_BACKEND_AUTOMATIC);
    uni_simd_symbol_sync4_block_t block{};
    block.descriptor_size = UNI_SIMD_SYMBOL_SYNC4_BLOCK_DESCRIPTOR_SIZE;
    uni_simd_symbol_sync4_result_t result{};
    result.descriptor_size = UNI_SIMD_SYMBOL_SYNC4_RESULT_DESCRIPTOR_SIZE;
    Check(uni_simd_kernel_execute(kernel, &block, &result) == UNI_SIMD_RESULT_INVALID_ARGUMENT, "omega_min below one is rejected");
    Check(uni_simd_kernel_free(kernel) == UNI_SIMD_RESULT_SUCCESS, "free");

    const auto valid_config = MakeConfig(settings);
    kernel = Create(valid_config, UNI_SIMD_BACKEND_AUTOMATIC);
    std::vector<Complex> buffer(1000U);
    for (std::size_t lane = 0U; lane < kLanes; ++lane) {
        block.input[lane] = reinterpret_cast<const float*>(buffer.data());
        block.input_count[lane] = 100U;
    }
    block.output[1] = reinterpret_cast<float*>(buffer.data() + 50);
    block.output_capacity[1] = 10U;
    Check(uni_simd_kernel_execute(kernel, &block, &result) == UNI_SIMD_RESULT_OVERLAPPING_BUFFERS, "output overlapping an input is rejected");

    std::array<std::vector<Complex>, kLanes> outputs;
    for (std::size_t lane = 0U; lane < kLanes; ++lane) {
        outputs[lane].resize(lane == 3U ? 5U : 100U);
        block.output[lane] = reinterpret_cast<float*>(outputs[lane].data());
        block.output_capacity[lane] = outputs[lane].size();
    }
    Check(uni_simd_kernel_execute(kernel, &block, &result) == UNI_SIMD_RESULT_SUCCESS, "execute");
    Check(result.truncated_mask == 8U && result.output_count[3] == 5U, "a full output truncates only its lane");
    Check(result.output_count[0] > 20U, "other lanes keep going");
    Check(uni_simd_kernel_free(kernel) == UNI_SIMD_RESULT_SUCCESS, "free");
}

} // namespace

int main() {
    Check(uni_simd_initialize() == UNI_SIMD_RESULT_SUCCESS, "initialize");
    for (const auto backend : {UNI_SIMD_BACKEND_GENERIC, UNI_SIMD_BACKEND_AUTOMATIC}) {
        TestMatchesOracle(backend, LoopSettings{}, "gardner/auto");
        TestMatchesOracle(backend, LoopSettings{.mm_threshold = 0.02, .fallback_threshold = 0.2}, "gardner switching to mm", 1U);
        TestMatchesOracle(backend, LoopSettings{.mm_threshold = 0.02, .fallback_threshold = 1e-5}, "switching back and forth", 2U);
        TestMatchesOracle(backend, LoopSettings{.auto_ted = false}, "gardner");
        TestMatchesOracle(backend, LoopSettings{.ted = UNI_SIMD_SYMBOL_SYNC_TED_MUELLER_MULLER}, "mm");
        TestMatchesOracle(backend, LoopSettings{.interpolator = UNI_SIMD_SYMBOL_SYNC_INTERPOLATOR_LINEAR}, "linear");
        TestMatchesOracle(backend, LoopSettings{.omega = 2.0, .deviation = 0.2, .loop_bw = 0.02}, "two samples per symbol");
    }
    TestBackendsAgreeAndSplitInvariant();
    TestAcquiresTimingAndSwitchesDetector();
    TestValidation();
    Check(uni_simd_finalize() == UNI_SIMD_RESULT_SUCCESS, "finalize");
    std::puts("symbol_sync4 tests passed");
    return 0;
}
