#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include <uni/simd/simd.hpp>

#include "cpu_topology.hpp"

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
namespace topology = uni::simd::benchmark::cpu_topology;

#if defined(__clang__) || defined(__GNUC__)
constexpr bool kOptimizedBuild =
#if defined(__OPTIMIZE__)
    true;
#else
    false;
#endif
#else
constexpr bool kOptimizedBuild =
#if defined(NDEBUG)
    true;
#else
    false;
#endif
#endif

constexpr std::array kCandidateBackends{
    uni::simd::Backend::generic,
    uni::simd::Backend::sse2,
    uni::simd::Backend::avx2,
    uni::simd::Backend::avx2_fma,
    uni::simd::Backend::avx512,
    uni::simd::Backend::automatic,
};
constexpr std::size_t kBackendSlots = static_cast<std::size_t>(uni::simd::Backend::neon) + 1U;

struct WorkloadProfile {
    std::string_view label;
    std::size_t working_set_bytes;
    std::size_t address_lanes;
    bool prewarm_samples;
};

constexpr std::array kWorkloadProfiles{
    WorkloadProfile{"16 KiB", 16U * 1024U, 16U, true},
    WorkloadProfile{"256 KiB", 256U * 1024U, 16U, true},
    WorkloadProfile{"4 MiB", 4U * 1024U * 1024U, 4U, true},
    WorkloadProfile{"32 MiB", 32U * 1024U * 1024U, 2U, true},
    WorkloadProfile{"256 MiB", 256U * 1024U * 1024U, 2U, false},
};
constexpr std::size_t kBufferAlignment = 128U;
constexpr std::size_t kProfileColumnWidth = 26U;
constexpr auto kCalibrationDuration = std::chrono::milliseconds{2};
volatile double checksum_sink = 0.0;

struct Config {
    std::size_t samples = 15U;
    std::size_t warmup_batches = 3U;
    std::size_t sample_milliseconds = 10U;
};

struct Statistics {
    double nanoseconds_per_item;
    double gibibytes_per_second;
    double relative_mad_percent;
};

template <typename Value>
class AlignedAllocator {
public:
    using value_type = Value;

    AlignedAllocator() noexcept = default;

    template <typename Other>
    constexpr AlignedAllocator(const AlignedAllocator<Other>&) noexcept {
    }

    [[nodiscard]] Value* allocate(const std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(Value)) {
            throw std::bad_array_new_length{};
        }
        return static_cast<Value*>(::operator new(count * sizeof(Value), std::align_val_t{kBufferAlignment}));
    }

    void deallocate(Value* const pointer, std::size_t) noexcept {
        ::operator delete(pointer, std::align_val_t{kBufferAlignment});
    }

    template <typename Other>
    struct rebind {
        using other = AlignedAllocator<Other>;
    };
};

template <typename Left, typename Right>
[[nodiscard]] constexpr bool operator==(const AlignedAllocator<Left>&, const AlignedAllocator<Right>&) noexcept {
    return true;
}

template <typename Value>
using AlignedBuffer = std::vector<Value, AlignedAllocator<Value>>;

template <typename Value>
using BufferLanes = std::vector<AlignedBuffer<Value>>;

template <typename Value>
struct alignas(kBufferAlignment) AlignedValue {
    Value value{};
};

enum class ParseResult {
    run,
    help,
    error,
};

#if defined(_WIN32)
class ConsolePauseOnExit final {
public:
    ConsolePauseOnExit() noexcept {
        std::array<DWORD, 2U> process_ids{};
        enabled_ = GetConsoleProcessList(process_ids.data(), static_cast<DWORD>(process_ids.size())) == 1U;
    }

    ~ConsolePauseOnExit() {
        if (enabled_) {
            std::cout << "\nPress Enter to exit..." << std::flush;
            (void)std::cin.get();
        }
    }

private:
    bool enabled_ = false;
};
#endif

[[nodiscard]] std::optional<std::size_t> parse_size(const std::string_view text) {
    std::uint64_t value = 0U;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0U ||
        value > std::numeric_limits<std::size_t>::max()) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(value);
}

void print_usage(const std::string_view program) {
    fmt::print("Usage: {} [--samples N] [--warmup N] [--sample-ms N]\n", program);
}

[[nodiscard]] ParseResult parse_arguments(const int argc, char** argv, Config& config) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        if (argument == "--help" || argument == "-h") {
            return ParseResult::help;
        }
        if (argument != "--samples" && argument != "--iterations" && argument != "--warmup" &&
            argument != "--sample-ms") {
            std::cerr << "Unknown argument: " << argument << '\n';
            return ParseResult::error;
        }
        if (++index == argc) {
            std::cerr << "Missing value for " << argument << '\n';
            return ParseResult::error;
        }

        const auto value = parse_size(argv[index]);
        if (!value.has_value()) {
            std::cerr << "Invalid positive integer for " << argument << ": " << argv[index] << '\n';
            return ParseResult::error;
        }
        if (argument == "--samples" || argument == "--iterations") {
            config.samples = *value;
        } else if (argument == "--warmup") {
            config.warmup_batches = *value;
        } else {
            config.sample_milliseconds = *value;
        }
    }
    return ParseResult::run;
}

void require_success(const uni::simd::Result result) {
    if (!uni::simd::succeeded(result)) {
        throw std::runtime_error("SIMD operation failed");
    }
}

[[nodiscard]] double median(std::vector<double> values) {
    if (values.empty()) {
        throw std::runtime_error("benchmark produced no samples");
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2U;
    if (values.size() % 2U != 0U) {
        return values[middle];
    }
    return (values[middle - 1U] + values[middle]) * 0.5;
}

template <typename Operation, typename Checksum>
[[nodiscard]] Statistics measure(const Config& config, const WorkloadProfile& profile, const std::size_t item_count,
                                 const double bytes_per_iteration, Operation&& operation, Checksum&& checksum) {
    std::size_t next_lane = 0U;
    const auto take_lane = [&] {
        const std::size_t lane = next_lane;
        next_lane = (next_lane + 1U) % profile.address_lanes;
        return lane;
    };
    const auto run_batch = [&](const std::size_t lane, const std::size_t repetitions) {
        for (std::size_t repetition = 0U; repetition < repetitions; ++repetition) {
            const std::size_t active_lane = profile.prewarm_samples
                                                ? lane
                                                : (lane + repetition) % profile.address_lanes;
            require_success(operation(active_lane));
        }
    };

    std::size_t calibration_repetitions = 1U;
    double calibration_seconds = 0.0;
    for (;;) {
        const std::size_t lane = take_lane();
        if (profile.prewarm_samples) {
            require_success(operation(lane));
        }
        const auto begin = Clock::now();
        run_batch(lane, calibration_repetitions);
        calibration_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
        if (calibration_seconds >= std::chrono::duration<double>(kCalibrationDuration).count()) {
            break;
        }
        if (calibration_repetitions > std::numeric_limits<std::size_t>::max() / 2U) {
            throw std::runtime_error("benchmark repetition count overflow");
        }
        calibration_repetitions *= 2U;
    }

    const double target_seconds = static_cast<double>(config.sample_milliseconds) / 1000.0;
    const double scaled_repetitions =
        std::ceil(static_cast<double>(calibration_repetitions) * target_seconds / calibration_seconds);
    if (scaled_repetitions > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("benchmark repetition count overflow");
    }
    const std::size_t batch_repetitions = std::max<std::size_t>(1U, static_cast<std::size_t>(scaled_repetitions));

    for (std::size_t iteration = 0U; iteration < config.warmup_batches; ++iteration) {
        run_batch(take_lane(), batch_repetitions);
    }

    std::vector<double> nanoseconds_per_item;
    nanoseconds_per_item.reserve(config.samples);
    std::size_t last_lane = 0U;
    for (std::size_t sample = 0U; sample < config.samples; ++sample) {
        last_lane = take_lane();
        if (profile.prewarm_samples) {
            require_success(operation(last_lane));
        }
        const auto begin = Clock::now();
        run_batch(last_lane, batch_repetitions);
        const double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
        if (seconds <= 0.0) {
            throw std::runtime_error("benchmark timer resolution is insufficient");
        }
        const double processed_items = static_cast<double>(item_count) * static_cast<double>(batch_repetitions);
        nanoseconds_per_item.push_back(seconds * 1.0e9 / processed_items);
    }

    checksum_sink = checksum(last_lane);
    const double median_nanoseconds = median(nanoseconds_per_item);
    std::vector<double> absolute_deviations;
    absolute_deviations.reserve(nanoseconds_per_item.size());
    for (const double sample : nanoseconds_per_item) {
        absolute_deviations.push_back(std::abs(sample - median_nanoseconds));
    }
    const double relative_mad_percent =
        median_nanoseconds == 0.0 ? 0.0 : 148.26 * median(std::move(absolute_deviations)) / median_nanoseconds;
    const double bytes_per_item = bytes_per_iteration / static_cast<double>(item_count);
    return {
        .nanoseconds_per_item = median_nanoseconds,
        .gibibytes_per_second = bytes_per_item * 1.0e9 / median_nanoseconds / static_cast<double>(1ULL << 30U),
        .relative_mad_percent = relative_mad_percent,
    };
}

template <typename Value>
[[nodiscard]] double checksum(const AlignedBuffer<Value>& values) {
    double sum = 0.0;
    for (const Value value : values) {
        sum += static_cast<double>(value);
    }
    return sum;
}

[[nodiscard]] double checksum(const std::complex<float> value) {
    return static_cast<double>(value.real()) + static_cast<double>(value.imag());
}

template <typename Value>
void verify_exact(const AlignedBuffer<Value>& expected, const AlignedBuffer<Value>& actual) {
    if (expected != actual) {
        throw std::runtime_error("backend result differs from the generic reference");
    }
}

void verify_floats(const std::span<const float> expected, const std::span<const float> actual,
                   const float relative_tolerance = 1.0e-5f) {
    if (expected.size() != actual.size()) {
        throw std::runtime_error("backend result has an unexpected size");
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const float tolerance = relative_tolerance * std::max(1.0f, std::abs(expected[index]));
        if (!std::isfinite(expected[index]) || !std::isfinite(actual[index]) ||
            std::abs(expected[index] - actual[index]) > tolerance) {
            throw std::runtime_error("backend result differs from the generic reference");
        }
    }
}

void verify_complex(const std::complex<float> expected, const std::complex<float> actual,
                    const std::size_t accumulated_items) {
    const float relative_tolerance = 1.0e-4f + 8.0f * std::numeric_limits<float>::epsilon() *
                                                       std::sqrt(static_cast<float>(accumulated_items));
    const float tolerance = relative_tolerance * std::max(1.0f, std::abs(expected));
    if (!std::isfinite(actual.real()) || !std::isfinite(actual.imag()) ||
        std::abs(expected - actual) > tolerance) {
        std::cerr << "complex result mismatch: expected (" << expected.real() << ", " << expected.imag()
                  << "), actual (" << actual.real() << ", " << actual.imag() << ")\n";
        throw std::runtime_error("backend result differs from the generic reference");
    }
}

[[nodiscard]] std::string table_border(const std::span<const std::size_t> widths, const char fill = '-') {
    std::string border{"+"};
    for (const std::size_t width : widths) {
        border += std::string(width + 2U, fill);
        border += '+';
    }
    return border;
}

[[nodiscard]] std::string outer_border(const std::string_view table_line) {
    return "+" + std::string(table_line.size() - 2U, '-') + "+";
}

[[nodiscard]] std::string fit_text(std::string value, const std::size_t width) {
    if (value.size() <= width) {
        return value;
    }
    if (width <= 3U) {
        return value.substr(0U, width);
    }
    value.resize(width - 3U);
    value += "...";
    return value;
}

[[nodiscard]] std::string core_type_name(const topology::CoreClass& core_class) {
    const std::string_view key = core_class.key;
    if (key.starts_with("amd_")) {
        const std::string_view architecture = key.substr(4U);
        return std::string(architecture.substr(0U, architecture.find('_')));
    }

    std::string_view label = core_class.label;
    if (const std::size_t separator = label.find('@'); separator != std::string_view::npos) {
        label = label.substr(0U, separator);
    }
    return label.empty() ? "default" : std::string(label);
}

struct ProfileMeasurement {
    Statistics statistics;
    std::optional<double> speedup;
};

struct ResultRow {
    std::string name;
    std::string backend;
    std::array<std::optional<ProfileMeasurement>, kWorkloadProfiles.size()> profiles;
};

class BenchmarkResults final {
public:
    void add(const std::string_view name, const std::string_view backend, const std::size_t profile_index,
             const Statistics& statistics, const std::optional<double> speedup) {
        auto row = std::find_if(rows_.begin(), rows_.end(), [&](const ResultRow& candidate) {
            return candidate.name == name && candidate.backend == backend;
        });
        if (row == rows_.end()) {
            rows_.push_back({.name = std::string(name), .backend = std::string(backend)});
            row = std::prev(rows_.end());
        }
        row->profiles.at(profile_index) = ProfileMeasurement{statistics, speedup};
    }

    void print(const topology::CoreClass& core_class, const topology::ThreadAffinityStatus affinity_status,
               const std::size_t class_index, const std::size_t class_count) const {
        constexpr auto widths = [] {
            std::array<std::size_t, kWorkloadProfiles.size() + 2U> result{};
            result[0] = 30U;
            result[1] = 8U;
            std::fill(result.begin() + 2, result.end(), kProfileColumnWidth);
            return result;
        }();
        const std::string border = table_border(widths);
        const std::string frame = outer_border(border);
        const std::string placement = affinity_status == topology::ThreadAffinityStatus::unsupported
                                          ? "scheduler-managed; affinity unsupported"
                                          : fmt::format("pinned to cpu{}", core_class.logical_processor.logical_processor_id);
        const std::string title = fmt::format("RESULTS {}/{}: {} ({})", class_index + 1U, class_count,
                                              core_type_name(core_class), placement);
        fmt::print("\n{}\n| {:^{}} |\n{}\n", frame, title, frame.size() - 4U, border);
        fmt::print("| {:<30} | {:<8} |", "kernel", "backend");
        for (const auto& profile : kWorkloadProfiles) {
            fmt::print(" {:^{}} |", profile.label, kProfileColumnWidth);
        }
        fmt::print("\n| {:<30} | {:<8} |", "", "");
        const std::string metric_header =
            fmt::format("{:>6} {:>5} {:>5} {:>7}", "ns/i", "GiB/s", "MAD%", "speedup");
        for (std::size_t index = 0U; index < kWorkloadProfiles.size(); ++index) {
            fmt::print(" {:>{}} |", metric_header, kProfileColumnWidth);
        }
        fmt::print("\n{}\n", table_border(widths, '='));

        std::string_view previous_name;
        for (const auto& row : rows_) {
            if (!previous_name.empty() && previous_name != row.name) {
                fmt::print("{}\n", border);
            }
            previous_name = row.name;
            fmt::print("| {:<30} | {:<8} |", row.name, row.backend);
            for (const auto& profile : row.profiles) {
                fmt::print(" {:>{}} |", format_profile(profile), kProfileColumnWidth);
            }
            fmt::print("\n");
        }
        fmt::print("{}\n", border);
    }

private:
    [[nodiscard]] static std::string format_profile(const std::optional<ProfileMeasurement>& measurement) {
        if (!measurement.has_value()) {
            return "-";
        }
        const std::string speedup =
            measurement->speedup.has_value() ? fmt::format("{:.2f}x", *measurement->speedup) : "-";
        const std::string mad = fmt::format("{:.1f}%", measurement->statistics.relative_mad_percent);
        return fmt::format("{:>6.3f} {:>5.1f} {:>5} {:>7}", measurement->statistics.nanoseconds_per_item,
                           measurement->statistics.gibibytes_per_second, mad, speedup);
    }

    std::vector<ResultRow> rows_;
};

class BenchmarkRunner final {
public:
    BenchmarkRunner(const Config& config, const WorkloadProfile& profile, const std::size_t profile_index,
                    BenchmarkResults& results)
        : config_(config), profile_(profile), profile_index_(profile_index), results_(results) {
    }

    template <typename Operation, typename Validator, typename Checksum>
    void run(const std::string_view name, const uni::simd::Kernel kernel, const std::size_t item_count,
             const double bytes_per_iteration, Operation&& operation, Validator&& validator,
             Checksum&& checksum_function) const {
        std::array<bool, kBackendSlots> measured_backends{};
        std::optional<double> generic_nanoseconds_per_item;
        for (const auto requested_backend : kCandidateBackends) {
            const auto context = uni::simd::create_context({.backend = requested_backend});
            if (!context.has_value()) {
                continue;
            }

            const auto backend = context->kernel_backend(kernel);
            const auto backend_index = static_cast<std::size_t>(backend);
            if (backend_index >= measured_backends.size() || measured_backends[backend_index]) {
                continue;
            }
            measured_backends[backend_index] = true;

            require_success(operation(*context, 0U));
            validator(0U);
            const Statistics statistics = measure(
                config_, profile_, item_count, bytes_per_iteration,
                [&](const std::size_t lane) { return operation(*context, lane); }, checksum_function);
            if (backend == uni::simd::Backend::generic) {
                generic_nanoseconds_per_item = statistics.nanoseconds_per_item;
            }
            std::optional<double> speedup;
            if (generic_nanoseconds_per_item.has_value()) {
                speedup = *generic_nanoseconds_per_item / statistics.nanoseconds_per_item;
            }
            results_.add(name, uni::simd::backend_name(backend), profile_index_, statistics, speedup);
        }
    }

    template <typename Operation, typename Validator, typename Checksum>
    void run_runtime(const std::string_view name, const std::size_t item_count, const double bytes_per_iteration,
                     Operation&& operation, Validator&& validator, Checksum&& checksum_function) const {
        require_success(operation(0U));
        validator(0U);
        const Statistics statistics =
            measure(config_, profile_, item_count, bytes_per_iteration, operation, checksum_function);
        results_.add(name, "runtime", profile_index_, statistics, std::nullopt);
    }

private:
    const Config& config_;
    const WorkloadProfile& profile_;
    std::size_t profile_index_;
    BenchmarkResults& results_;
};

template <typename Value>
[[nodiscard]] BufferLanes<Value> make_lanes(const std::size_t lane_count, const std::size_t item_count) {
    BufferLanes<Value> lanes;
    lanes.reserve(lane_count);
    for (std::size_t lane = 0U; lane < lane_count; ++lane) {
        lanes.emplace_back(item_count);
        if (reinterpret_cast<std::uintptr_t>(lanes.back().data()) % kBufferAlignment != 0U) {
            throw std::runtime_error("benchmark buffer alignment is insufficient");
        }
    }
    return lanes;
}

void fill_bytes(AlignedBuffer<std::uint8_t>& bytes) {
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::uint8_t>((index * 37U + 11U) & 0xffU);
    }
}

void fill_bits(AlignedBuffer<std::uint8_t>& bits) {
    for (std::size_t index = 0; index < bits.size(); ++index) {
        bits[index] = static_cast<std::uint8_t>((index * 5U + 1U) & 1U);
    }
}

void fill_complex(AlignedBuffer<std::complex<float>>& complex_values) {
    for (std::size_t index = 0; index < complex_values.size(); ++index) {
        complex_values[index] = {
            static_cast<float>(index % 127U) / 63.5f - 1.0f,
            static_cast<float>(index % 61U) / 30.5f - 1.0f,
        };
    }
}

void fill_taps(AlignedBuffer<float>& taps) {
    for (std::size_t index = 0; index < taps.size(); ++index) {
        taps[index] = static_cast<float>(static_cast<int>(index % 31U) - 15) / 31.0f;
    }
}

void run_benchmark_profile(const Config& config, const WorkloadProfile& profile, const std::size_t profile_index,
                           BenchmarkResults& results) {
    const std::size_t byte_count = std::max<std::size_t>(64U, profile.working_set_bytes / 2U);
    const std::size_t requested_bits = std::max<std::size_t>(64U, profile.working_set_bytes * 8U / 9U);
    const std::size_t bit_count = requested_bits - requested_bits % 8U;
    const std::size_t packed_count = bit_count / 8U;
    const std::size_t quantized_count = std::max<std::size_t>(8U, profile.working_set_bytes / 10U);
    const std::size_t magnitude_count = std::max<std::size_t>(8U, profile.working_set_bytes / 12U);
    const std::size_t dot_count = magnitude_count;
    std::size_t symmetric_count = std::max<std::size_t>(9U, profile.working_set_bytes / 10U);
    if (symmetric_count % 2U == 0U) {
        --symmetric_count;
    }
    const std::size_t tap_pair_count = (symmetric_count - 1U) / 2U;
    const auto generic = uni::simd::create_context({.backend = uni::simd::Backend::generic});
    if (!generic.has_value()) {
        throw std::runtime_error("generic backend is unavailable");
    }

    const double byte_items = static_cast<double>(byte_count);
    const double bit_items = static_cast<double>(bit_count);
    const double magnitude_items = static_cast<double>(magnitude_count);
    BenchmarkRunner runner(config, profile, profile_index, results);

    {
        auto inputs = make_lanes<std::uint8_t>(profile.address_lanes, byte_count);
        auto outputs = make_lanes<std::uint8_t>(profile.address_lanes, byte_count);
        for (auto& input : inputs) {
            fill_bytes(input);
        }

        runner.run_runtime(
            "copy", byte_count, byte_items * 2.0,
            [&](const std::size_t lane) { return generic->copy(outputs[lane], inputs[lane]); },
            [&](const std::size_t lane) { verify_exact(inputs[lane], outputs[lane]); },
            [&](const std::size_t lane) { return checksum(outputs[lane]); });

        AlignedBuffer<std::uint8_t> reference(byte_count);
        require_success(generic->invert_lsb(reference, inputs[0]));
        runner.run(
            "invert_lsb", uni::simd::Kernel::invert_lsb, byte_count, byte_items * 2.0,
            [&](const uni::simd::Context& context, const std::size_t lane) {
                return context.invert_lsb(outputs[lane], inputs[lane]);
            },
            [&](const std::size_t lane) { verify_exact(reference, outputs[lane]); },
            [&](const std::size_t lane) { return checksum(outputs[lane]); });

        require_success(generic->invert_bytes(reference, inputs[0]));
        runner.run(
            "invert_bytes", uni::simd::Kernel::invert_bytes, byte_count, byte_items * 2.0,
            [&](const uni::simd::Context& context, const std::size_t lane) {
                return context.invert_bytes(outputs[lane], inputs[lane]);
            },
            [&](const std::size_t lane) { verify_exact(reference, outputs[lane]); },
            [&](const std::size_t lane) { return checksum(outputs[lane]); });
    }

    {
        auto inputs = make_lanes<std::uint8_t>(profile.address_lanes, bit_count);
        auto outputs = make_lanes<std::uint8_t>(profile.address_lanes, packed_count);
        for (auto& input : inputs) {
            fill_bits(input);
        }
        AlignedBuffer<std::uint8_t> reference(packed_count);

        require_success(generic->pack_bits_lsb(reference, inputs[0]));
        runner.run(
            "pack_bits_lsb", uni::simd::Kernel::pack_bits_lsb, bit_count, bit_items + static_cast<double>(packed_count),
            [&](const uni::simd::Context& context, const std::size_t lane) {
                return context.pack_bits_lsb(outputs[lane], inputs[lane]);
            },
            [&](const std::size_t lane) { verify_exact(reference, outputs[lane]); },
            [&](const std::size_t lane) { return checksum(outputs[lane]); });

        require_success(generic->pack_bits_msb(reference, inputs[0]));
        runner.run(
            "pack_bits_msb", uni::simd::Kernel::pack_bits_msb, bit_count, bit_items + static_cast<double>(packed_count),
            [&](const uni::simd::Context& context, const std::size_t lane) {
                return context.pack_bits_msb(outputs[lane], inputs[lane]);
            },
            [&](const std::size_t lane) { verify_exact(reference, outputs[lane]); },
            [&](const std::size_t lane) { return checksum(outputs[lane]); });
    }

    const auto run_unpack = [&](const std::string_view name, const uni::simd::Kernel kernel, const bool msb) {
        AlignedBuffer<std::uint8_t> bits(bit_count);
        fill_bits(bits);
        AlignedBuffer<std::uint8_t> packed(packed_count);
        require_success(msb ? generic->pack_bits_msb(packed, bits) : generic->pack_bits_lsb(packed, bits));
        auto inputs = make_lanes<std::uint8_t>(profile.address_lanes, packed_count);
        auto outputs = make_lanes<std::uint8_t>(profile.address_lanes, bit_count);
        for (auto& input : inputs) {
            input = packed;
        }
        AlignedBuffer<std::uint8_t> reference(bit_count);
        require_success(msb ? generic->unpack_bits_msb(reference, packed)
                            : generic->unpack_bits_lsb(reference, packed));
        runner.run(
            name, kernel, bit_count, bit_items + static_cast<double>(packed_count),
            [&](const uni::simd::Context& context, const std::size_t lane) {
                return msb ? context.unpack_bits_msb(outputs[lane], inputs[lane])
                           : context.unpack_bits_lsb(outputs[lane], inputs[lane]);
            },
            [&](const std::size_t lane) { verify_exact(reference, outputs[lane]); },
            [&](const std::size_t lane) { return checksum(outputs[lane]); });
    };
    run_unpack("unpack_bits_lsb", uni::simd::Kernel::unpack_bits_lsb, false);
    run_unpack("unpack_bits_msb", uni::simd::Kernel::unpack_bits_msb, true);

    {
        auto inputs = make_lanes<std::complex<float>>(profile.address_lanes, quantized_count);
        auto outputs = make_lanes<std::uint8_t>(profile.address_lanes, quantized_count * 2U);
        for (auto& input : inputs) {
            fill_complex(input);
        }
        AlignedBuffer<std::uint8_t> reference(quantized_count * 2U);
        require_success(generic->quantize_interleaved_cf32_u8(reference, inputs[0], {.scale = -7.0f}));
        runner.run(
            "quantize_interleaved_cf32_u8", uni::simd::Kernel::quantize_interleaved_cf32_u8, quantized_count,
            static_cast<double>(quantized_count) * 10.0,
            [&](const uni::simd::Context& context, const std::size_t lane) {
                return context.quantize_interleaved_cf32_u8(outputs[lane], inputs[lane], {.scale = -7.0f});
            },
            [&](const std::size_t lane) { verify_exact(reference, outputs[lane]); },
            [&](const std::size_t lane) { return checksum(outputs[lane]); });
    }

    {
        auto inputs = make_lanes<std::complex<float>>(profile.address_lanes, magnitude_count);
        auto outputs = make_lanes<float>(profile.address_lanes, magnitude_count);
        for (auto& input : inputs) {
            fill_complex(input);
        }
        AlignedBuffer<float> reference(magnitude_count);

        require_success(generic->magnitude_squared(reference, inputs[0], 3.0f));
        runner.run(
            "magnitude_squared_cf32", uni::simd::Kernel::magnitude_squared_cf32, magnitude_count,
            magnitude_items * 12.0,
            [&](const uni::simd::Context& context, const std::size_t lane) {
                return context.magnitude_squared(outputs[lane], inputs[lane], 3.0f);
            },
            [&](const std::size_t lane) { verify_floats(reference, outputs[lane]); },
            [&](const std::size_t lane) { return checksum(outputs[lane]); });

        require_success(generic->power_spectral_density(reference, inputs[0], 3.0f, 2.0f));
        // PSD uses the same backend selection as magnitude_squared, but has no
        // separate Kernel enum value.
        runner.run(
            "power_spectral_density_cf32", uni::simd::Kernel::magnitude_squared_cf32, magnitude_count,
            magnitude_items * 12.0,
            [&](const uni::simd::Context& context, const std::size_t lane) {
                return context.power_spectral_density(outputs[lane], inputs[lane], 3.0f, 2.0f);
            },
            [&](const std::size_t lane) { verify_floats(reference, outputs[lane]); },
            [&](const std::size_t lane) { return checksum(outputs[lane]); });
    }

    {
        auto inputs = make_lanes<std::complex<float>>(profile.address_lanes, dot_count);
        auto taps = make_lanes<float>(profile.address_lanes, dot_count);
        for (auto& input : inputs) {
            fill_complex(input);
        }
        for (auto& lane_taps : taps) {
            fill_taps(lane_taps);
        }
        AlignedValue<std::complex<float>> reference;
        AlignedBuffer<AlignedValue<std::complex<float>>> outputs(profile.address_lanes);
        require_success(generic->dot_cf32_f32(reference.value, inputs[0], taps[0]));
        runner.run(
            "dot_cf32_f32", uni::simd::Kernel::dot_cf32_f32, dot_count,
            static_cast<double>(dot_count) * 12.0 + static_cast<double>(sizeof(std::complex<float>)),
            [&](const uni::simd::Context& context, const std::size_t lane) {
                return context.dot_cf32_f32(outputs[lane].value, inputs[lane], taps[lane]);
            },
            [&](const std::size_t lane) { verify_complex(reference.value, outputs[lane].value, dot_count); },
            [&](const std::size_t lane) { return checksum(outputs[lane].value); });
    }

    const double symmetric_bytes = static_cast<double>(symmetric_count) * sizeof(std::complex<float>) +
                                   static_cast<double>(tap_pair_count) * sizeof(float) + sizeof(float) +
                                   sizeof(std::complex<float>);
    {
        auto inputs = make_lanes<std::complex<float>>(profile.address_lanes, symmetric_count);
        auto taps = make_lanes<float>(profile.address_lanes, tap_pair_count);
        for (auto& input : inputs) {
            fill_complex(input);
        }
        for (auto& lane_taps : taps) {
            fill_taps(lane_taps);
        }
        AlignedValue<std::complex<float>> reference;
        AlignedBuffer<AlignedValue<std::complex<float>>> outputs(profile.address_lanes);
        require_success(generic->dot_symmetric_cf32_f32(reference.value, inputs[0], taps[0], 0.25f));
        runner.run(
            "dot_symmetric_cf32_f32", uni::simd::Kernel::dot_symmetric_cf32_f32, symmetric_count, symmetric_bytes,
            [&](const uni::simd::Context& context, const std::size_t lane) {
                return context.dot_symmetric_cf32_f32(outputs[lane].value, inputs[lane], taps[lane], 0.25f);
            },
            [&](const std::size_t lane) { verify_complex(reference.value, outputs[lane].value, symmetric_count); },
            [&](const std::size_t lane) { return checksum(outputs[lane].value); });
    }
}

void run_benchmarks_on_core(const Config& config, const topology::CoreClass& core_class,
                            const topology::ThreadAffinityStatus affinity_status, const std::size_t class_index,
                            const std::size_t class_count) {
    BenchmarkResults results;
    for (std::size_t profile_index = 0U; profile_index < kWorkloadProfiles.size(); ++profile_index) {
        run_benchmark_profile(config, kWorkloadProfiles[profile_index], profile_index, results);
    }
    results.print(core_class, affinity_status, class_index, class_count);
}

[[nodiscard]] std::string collect_members(const topology::Snapshot& snapshot, const std::string_view class_key) {
    std::vector<int> members;
    for (const auto& processor : snapshot.logical_processors) {
        const std::string_view key =
            processor.performance_class_key.empty() ? std::string_view{"default"} : processor.performance_class_key;
        if (key == class_key) {
            members.push_back(processor.logical_processor_id);
        }
    }
    if (members.empty()) {
        return "n/a";
    }

    std::sort(members.begin(), members.end());
    std::string result;
    const auto append_range = [&](const int first, const int last) {
        if (!result.empty()) {
            result += ',';
        }
        result += first == last ? std::to_string(first) : fmt::format("{}-{}", first, last);
    };
    int first = members.front();
    int previous = first;
    for (const int cpu : std::span<const int>{members}.subspan(1U)) {
        if (cpu != previous + 1) {
            append_range(first, previous);
            first = cpu;
        }
        previous = cpu;
    }
    append_range(first, previous);
    return result;
}

void print_topology_table(const Config& config, const topology::Result& result,
                          const std::vector<topology::CoreClass>& core_classes) {
    constexpr std::array<std::size_t, 2U> summary_widths{22U, 79U};
    const std::string summary_border = table_border(summary_widths);
    const std::string summary_frame = outer_border(summary_border);
    const auto print_summary_row = [](const std::string_view name, std::string value) {
        fmt::print("| {:<22} | {:<79} |\n", name, fit_text(std::move(value), 79U));
    };
    std::string working_sets;
    std::string address_lanes;
    for (const auto& profile : kWorkloadProfiles) {
        if (!working_sets.empty()) {
            working_sets += " / ";
            address_lanes += " / ";
        }
        working_sets += profile.label;
        address_lanes += std::to_string(profile.address_lanes);
    }

    fmt::print("{}\n| {:^{}} |\n{}\n", summary_frame, "CPU TOPOLOGY", summary_frame.size() - 4U, summary_border);
    print_summary_row("Model", result.snapshot.model_name);
    print_summary_row("Processors", fmt::format("{} logical / {} physical", result.snapshot.logical_processor_count,
                                                result.snapshot.physical_core_count));
    print_summary_row("Packages / NUMA nodes",
                      fmt::format("{} / {}", result.snapshot.package_count, result.snapshot.numa_node_count));
    print_summary_row("Core types", std::to_string(core_classes.size()));
    print_summary_row("Working sets", std::move(working_sets));
    print_summary_row("Measurement", fmt::format("samples={}, warmup batches={}, target={} ms", config.samples,
                                                 config.warmup_batches, config.sample_milliseconds));
    print_summary_row("Buffers", fmt::format("{}-byte aligned; address lanes={}", kBufferAlignment, address_lanes));
    print_summary_row("Cache policy", "rotate addresses; prewarm through 32 MiB; stream 256 MiB");
    fmt::print("{}\n\n", summary_border);

    constexpr std::array<std::size_t, 8U> core_widths{3U, 10U, 7U, 31U, 7U, 7U, 7U, 7U};
    const std::string core_border = table_border(core_widths);
    const std::string core_frame = outer_border(core_border);
    fmt::print("{}\n| {:^{}} |\n{}\n", core_frame, "CORE TYPES", core_frame.size() - 4U, core_border);
    fmt::print("| {:>3} | {:<10} | {:>7} | {:<31} | {:>7} | {:>7} | {:>7} | {:>7} |\n", "#", "type", "pin CPU",
               "logical CPU members", "L1d KiB", "L2 KiB", "L3 MiB", "GHz");
    fmt::print("{}\n", core_border);
    for (std::size_t index = 0U; index < core_classes.size(); ++index) {
        const auto& core_class = core_classes[index];
        const auto& processor = core_class.logical_processor;
        const auto cache_kib = [](const std::uint64_t bytes) {
            return bytes == 0U ? std::string{"n/a"} : fmt::format("{}", bytes / 1024U);
        };
        const std::string l1 = cache_kib(processor.l1_data_cache_bytes);
        const std::string l2 = cache_kib(processor.l2_cache_bytes);
        const std::string l3 =
            processor.l3_cache_bytes == 0U
                ? "n/a"
                : fmt::format("{:.1f}", static_cast<double>(processor.l3_cache_bytes) / (1024.0 * 1024.0));
        const std::string frequency =
            processor.max_frequency_khz == 0U
                ? "n/a"
                : fmt::format("{:.2f}", static_cast<double>(processor.max_frequency_khz) / 1'000'000.0);
        fmt::print("| {:>3} | {:<10} | {:>7} | {:<31} | {:>7} | {:>7} | {:>7} | {:>7} |\n", index + 1U,
                   fit_text(core_type_name(core_class), 10U), processor.logical_processor_id,
                   fit_text(collect_members(result.snapshot, core_class.key), 31U), l1, l2, l3, frequency);
    }
    fmt::print("{}\n", core_border);
}

void run_benchmarks(const Config& config) {
    const topology::Result topology_result = topology::query_snapshot();
    if (!topology_result.status.has_data()) {
        throw std::runtime_error("CPU topology is unavailable: " + topology_result.status.message);
    }
    const std::vector<topology::CoreClass> core_classes = topology::build_core_classes(topology_result.snapshot);
    if (core_classes.empty()) {
        throw std::runtime_error("CPU topology did not provide a pinnable core class");
    }

    print_topology_table(config, topology_result, core_classes);

    for (std::size_t index = 0U; index < core_classes.size(); ++index) {
        const auto& core_class = core_classes[index];
        topology::ScopedThreadAffinity affinity(core_class.logical_processor);
        if (!affinity.can_run()) {
            throw std::runtime_error("failed to apply benchmark thread affinity for logical CPU " +
                                     std::to_string(core_class.logical_processor.logical_processor_id));
        }
        run_benchmarks_on_core(config, core_class, affinity.status(), index, core_classes.size());
    }
}

} // namespace

int main(const int argc, char** argv) {
#if defined(_WIN32)
    const ConsolePauseOnExit pause_on_exit;
#endif
    Config config;
    const ParseResult parse_result = parse_arguments(argc, argv, config);
    if (parse_result == ParseResult::help) {
        print_usage(argv[0]);
        return 0;
    }
    if (parse_result == ParseResult::error) {
        print_usage(argv[0]);
        return 2;
    }
    if (!kOptimizedBuild) {
        std::cerr << "Benchmark requires an optimized build; configure with -DCMAKE_BUILD_TYPE=Release\n";
        return 2;
    }

    try {
        run_benchmarks(config);
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
