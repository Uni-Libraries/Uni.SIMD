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
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include <uni/simd/simd.hpp>

#include "cpu_topology.hpp"

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
};

constexpr std::array kWorkloadProfiles{
    WorkloadProfile{"L1 16 KiB", 16U * 1024U},
    WorkloadProfile{"L2 512 KiB", 512U * 1024U},
    WorkloadProfile{"L3 4 MiB", 4U * 1024U * 1024U},
    WorkloadProfile{"large 64 MiB", 64U * 1024U * 1024U},
};
constexpr std::size_t kMinimumBatchBytes = 1U * 1024U * 1024U;
volatile double checksum_sink = 0.0;

struct Config {
    std::size_t iterations = 100U;
    std::size_t warmup_iterations = 10U;
};

struct Statistics {
    double nanoseconds_per_item;
    double gibibytes_per_second;
};

enum class ParseResult {
    run,
    help,
    error,
};

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
    fmt::print("Usage: {} [--iterations N] [--warmup N]\n", program);
}

[[nodiscard]] ParseResult parse_arguments(const int argc, char** argv, Config& config) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        if (argument == "--help" || argument == "-h") {
            return ParseResult::help;
        }
        if (argument != "--iterations" && argument != "--warmup") {
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
        if (argument == "--iterations") {
            config.iterations = *value;
        } else {
            config.warmup_iterations = *value;
        }
    }
    return ParseResult::run;
}

void require_success(const uni::simd::Result result) {
    if (!uni::simd::succeeded(result)) {
        throw std::runtime_error("SIMD operation failed");
    }
}

template <typename Operation, typename Checksum>
[[nodiscard]] Statistics measure(const Config& config, const std::size_t item_count,
                                  const double bytes_per_iteration, const std::size_t batch_repetitions,
                                  Operation&& operation, Checksum&& checksum) {
    for (std::size_t iteration = 0; iteration < config.warmup_iterations; ++iteration) {
        for (std::size_t repetition = 0; repetition < batch_repetitions; ++repetition) {
            require_success(operation());
        }
    }

    const auto begin = Clock::now();
    for (std::size_t iteration = 0; iteration < config.iterations; ++iteration) {
        for (std::size_t repetition = 0; repetition < batch_repetitions; ++repetition) {
            require_success(operation());
        }
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
    if (seconds <= 0.0) {
        throw std::runtime_error("benchmark timer resolution is insufficient");
    }

    const double total_repetitions = static_cast<double>(config.iterations) * static_cast<double>(batch_repetitions);
    const double processed_items = static_cast<double>(item_count) * total_repetitions;
    const double processed_gibibytes = bytes_per_iteration * total_repetitions /
                                        static_cast<double>(1ULL << 30U);
    checksum_sink = checksum();
    return {
        .nanoseconds_per_item = seconds * 1.0e9 / processed_items,
        .gibibytes_per_second = processed_gibibytes / seconds,
    };
}

template <typename Value>
[[nodiscard]] double checksum(const std::vector<Value>& values) {
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
void verify_exact(const std::vector<Value>& expected, const std::vector<Value>& actual) {
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
    const float real_tolerance = relative_tolerance * std::max(1.0f, std::abs(expected.real()));
    const float imag_tolerance = relative_tolerance * std::max(1.0f, std::abs(expected.imag()));
    if (!std::isfinite(actual.real()) || !std::isfinite(actual.imag()) ||
        std::abs(expected.real() - actual.real()) > real_tolerance ||
        std::abs(expected.imag() - actual.imag()) > imag_tolerance) {
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

    void print(const topology::CoreClass& core_class, const std::size_t class_index,
               const std::size_t class_count) const {
        constexpr std::array<std::size_t, 6U> widths{30U, 8U, 21U, 21U, 21U, 21U};
        const std::string border = table_border(widths);
        const std::string frame = outer_border(border);
        const std::string title = fmt::format("RESULTS {}/{}: {} (pinned to cpu{})", class_index + 1U, class_count,
                                              core_type_name(core_class),
                                              core_class.logical_processor.logical_processor_id);
        fmt::print("\n{}\n| {:^{}} |\n{}\n", frame, title, frame.size() - 4U, border);
        fmt::print("| {:<30} | {:<8} |", "kernel", "backend");
        for (const auto& profile : kWorkloadProfiles) {
            fmt::print(" {:^21} |", profile.label);
        }
        fmt::print("\n| {:<30} | {:<8} |", "", "");
        for (std::size_t index = 0U; index < kWorkloadProfiles.size(); ++index) {
            fmt::print(" {:^21} |", "ns/i GiB/s vs generic");
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
                fmt::print(" {:>21} |", format_profile(profile));
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
        const std::string speedup = measurement->speedup.has_value()
                                        ? fmt::format("{:.2f}x", *measurement->speedup)
                                        : "-";
        return fmt::format("{:>6.3f} {:>6.1f} {:>6}", measurement->statistics.nanoseconds_per_item,
                           measurement->statistics.gibibytes_per_second, speedup);
    }

    std::vector<ResultRow> rows_;
};

class BenchmarkRunner final {
public:
    BenchmarkRunner(const Config& config, const std::size_t profile_index, BenchmarkResults& results)
        : config_(config), profile_index_(profile_index), results_(results) {}

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

            require_success(operation(*context));
            validator();
            const Statistics statistics =
                measure(config_, item_count, bytes_per_iteration, batch_repetitions(bytes_per_iteration),
                        [&] { return operation(*context); }, checksum_function);
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
        require_success(operation());
        validator();
        const Statistics statistics = measure(config_, item_count, bytes_per_iteration,
                                              batch_repetitions(bytes_per_iteration), operation, checksum_function);
        results_.add(name, "runtime", profile_index_, statistics, std::nullopt);
    }

private:
    [[nodiscard]] static std::size_t batch_repetitions(const double bytes_per_iteration) {
        const auto bytes = static_cast<std::size_t>(bytes_per_iteration);
        return bytes >= kMinimumBatchBytes ? 1U : (kMinimumBatchBytes + bytes - 1U) / bytes;
    }

    const Config& config_;
    std::size_t profile_index_;
    BenchmarkResults& results_;
};

void fill_bytes(std::vector<std::uint8_t>& bytes) {
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::uint8_t>((index * 37U + 11U) & 0xffU);
    }
}

void fill_bits(std::vector<std::uint8_t>& bits) {
    for (std::size_t index = 0; index < bits.size(); ++index) {
        bits[index] = static_cast<std::uint8_t>((index * 5U + 1U) & 1U);
    }
}

void fill_numeric_inputs(std::vector<std::complex<float>>& complex_values, std::vector<float>& taps) {
    for (std::size_t index = 0; index < complex_values.size(); ++index) {
        complex_values[index] = {
            static_cast<float>(index % 127U) / 63.5f - 1.0f,
            static_cast<float>(index % 61U) / 30.5f - 1.0f,
        };
    }
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
    const std::size_t complex_count = std::max({quantized_count, magnitude_count, dot_count, symmetric_count});
    const std::size_t tap_count = std::max(dot_count, tap_pair_count);

    std::vector<std::uint8_t> byte_input(byte_count);
    std::vector<std::uint8_t> bit_input(bit_count);
    std::vector<std::complex<float>> complex_input(complex_count);
    std::vector<float> taps(tap_count);
    fill_bytes(byte_input);
    fill_bits(bit_input);
    fill_numeric_inputs(complex_input, taps);

    std::vector<std::uint8_t> invert_lsb_reference(byte_count);
    std::vector<std::uint8_t> invert_bytes_reference(byte_count);
    std::vector<std::uint8_t> packed_lsb_reference(packed_count);
    std::vector<std::uint8_t> packed_msb_reference(packed_count);
    std::vector<std::uint8_t> unpacked_lsb_reference(bit_count);
    std::vector<std::uint8_t> unpacked_msb_reference(bit_count);
    std::vector<std::uint8_t> quantized_reference(quantized_count * 2U);
    std::vector<float> magnitude_reference(magnitude_count);
    std::vector<float> psd_reference(magnitude_count);
    std::complex<float> dot_reference{};
    std::complex<float> symmetric_dot_reference{};

    const auto quantized_input = std::span<const std::complex<float>>{complex_input}.first(quantized_count);
    const auto magnitude_input = std::span<const std::complex<float>>{complex_input}.first(magnitude_count);
    const auto dot_input = std::span<const std::complex<float>>{complex_input}.first(dot_count);
    const auto dot_taps = std::span<const float>{taps}.first(dot_count);
    const auto symmetric_input = std::span<const std::complex<float>>{complex_input}.first(symmetric_count);
    const auto tap_pairs = std::span<const float>{taps}.first(tap_pair_count);

    const auto generic = uni::simd::create_context({.backend = uni::simd::Backend::generic});
    if (!generic.has_value()) {
        throw std::runtime_error("generic backend is unavailable");
    }
    require_success(generic->invert_lsb(invert_lsb_reference, byte_input));
    require_success(generic->invert_bytes(invert_bytes_reference, byte_input));
    require_success(generic->pack_bits_lsb(packed_lsb_reference, bit_input));
    require_success(generic->pack_bits_msb(packed_msb_reference, bit_input));
    require_success(generic->unpack_bits_lsb(unpacked_lsb_reference, packed_lsb_reference));
    require_success(generic->unpack_bits_msb(unpacked_msb_reference, packed_msb_reference));
    require_success(generic->quantize_interleaved_cf32_u8(quantized_reference, quantized_input, {.scale = -7.0f}));
    require_success(generic->magnitude_squared(magnitude_reference, magnitude_input, 3.0f));
    require_success(generic->power_spectral_density(psd_reference, magnitude_input, 3.0f, 2.0f));
    require_success(generic->dot_cf32_f32(dot_reference, dot_input, dot_taps));
    require_success(generic->dot_symmetric_cf32_f32(symmetric_dot_reference, symmetric_input, tap_pairs, 0.25f));

    std::vector<std::uint8_t> byte_output(byte_count);
    std::vector<std::uint8_t> packed_output(packed_count);
    std::vector<std::uint8_t> unpacked_output(bit_count);
    std::vector<std::uint8_t> quantized_output(quantized_count * 2U);
    std::vector<float> float_output(magnitude_count);
    std::complex<float> complex_output{};

    const double byte_items = static_cast<double>(byte_count);
    const double bit_items = static_cast<double>(bit_count);
    const double magnitude_items = static_cast<double>(magnitude_count);
    BenchmarkRunner runner(config, profile_index, results);

    runner.run_runtime(
        "copy", byte_count, byte_items * 2.0,
        [&] { return generic->copy(byte_output, byte_input); },
        [&] { verify_exact(byte_input, byte_output); }, [&] { return checksum(byte_output); });

    runner.run(
        "invert_lsb", uni::simd::Kernel::invert_lsb, byte_count, byte_items * 2.0,
        [&](const uni::simd::Context& context) { return context.invert_lsb(byte_output, byte_input); },
        [&] { verify_exact(invert_lsb_reference, byte_output); }, [&] { return checksum(byte_output); });

    runner.run(
        "invert_bytes", uni::simd::Kernel::invert_bytes, byte_count, byte_items * 2.0,
        [&](const uni::simd::Context& context) { return context.invert_bytes(byte_output, byte_input); },
        [&] { verify_exact(invert_bytes_reference, byte_output); }, [&] { return checksum(byte_output); });

    runner.run(
        "pack_bits_lsb", uni::simd::Kernel::pack_bits_lsb, bit_count, bit_items + static_cast<double>(packed_count),
        [&](const uni::simd::Context& context) { return context.pack_bits_lsb(packed_output, bit_input); },
        [&] { verify_exact(packed_lsb_reference, packed_output); }, [&] { return checksum(packed_output); });

    runner.run(
        "pack_bits_msb", uni::simd::Kernel::pack_bits_msb, bit_count, bit_items + static_cast<double>(packed_count),
        [&](const uni::simd::Context& context) { return context.pack_bits_msb(packed_output, bit_input); },
        [&] { verify_exact(packed_msb_reference, packed_output); }, [&] { return checksum(packed_output); });

    runner.run(
        "unpack_bits_lsb", uni::simd::Kernel::unpack_bits_lsb, bit_count, bit_items + static_cast<double>(packed_count),
        [&](const uni::simd::Context& context) { return context.unpack_bits_lsb(unpacked_output, packed_lsb_reference); },
        [&] { verify_exact(unpacked_lsb_reference, unpacked_output); }, [&] { return checksum(unpacked_output); });

    runner.run(
        "unpack_bits_msb", uni::simd::Kernel::unpack_bits_msb, bit_count, bit_items + static_cast<double>(packed_count),
        [&](const uni::simd::Context& context) { return context.unpack_bits_msb(unpacked_output, packed_msb_reference); },
        [&] { verify_exact(unpacked_msb_reference, unpacked_output); }, [&] { return checksum(unpacked_output); });

    runner.run(
        "quantize_interleaved_cf32_u8", uni::simd::Kernel::quantize_interleaved_cf32_u8, quantized_count,
        static_cast<double>(quantized_count) * 10.0,
        [&](const uni::simd::Context& context) {
            return context.quantize_interleaved_cf32_u8(quantized_output, quantized_input, {.scale = -7.0f});
        },
        [&] { verify_exact(quantized_reference, quantized_output); }, [&] { return checksum(quantized_output); });

    runner.run(
        "magnitude_squared_cf32", uni::simd::Kernel::magnitude_squared_cf32, magnitude_count,
        magnitude_items * 12.0,
        [&](const uni::simd::Context& context) { return context.magnitude_squared(float_output, magnitude_input, 3.0f); },
        [&] { verify_floats(magnitude_reference, float_output); }, [&] { return checksum(float_output); });

    // PSD uses the same backend selection as magnitude_squared, but has no separate Kernel enum value.
    runner.run(
        "power_spectral_density_cf32", uni::simd::Kernel::magnitude_squared_cf32, magnitude_count,
        magnitude_items * 12.0,
        [&](const uni::simd::Context& context) {
            return context.power_spectral_density(float_output, magnitude_input, 3.0f, 2.0f);
        },
        [&] { verify_floats(psd_reference, float_output); }, [&] { return checksum(float_output); });

    runner.run(
        "dot_cf32_f32", uni::simd::Kernel::dot_cf32_f32, dot_count,
        static_cast<double>(dot_count) * 12.0 + static_cast<double>(sizeof(std::complex<float>)),
        [&](const uni::simd::Context& context) { return context.dot_cf32_f32(complex_output, dot_input, dot_taps); },
        [&] { verify_complex(dot_reference, complex_output, dot_count); }, [&] { return checksum(complex_output); });

    const double symmetric_bytes = static_cast<double>(symmetric_count) * sizeof(std::complex<float>) +
                                   static_cast<double>(tap_pair_count) * sizeof(float) + sizeof(float) +
                                   sizeof(std::complex<float>);
    runner.run(
        "dot_symmetric_cf32_f32", uni::simd::Kernel::dot_symmetric_cf32_f32, symmetric_count, symmetric_bytes,
        [&](const uni::simd::Context& context) {
            return context.dot_symmetric_cf32_f32(complex_output, symmetric_input, tap_pairs, 0.25f);
        },
        [&] { verify_complex(symmetric_dot_reference, complex_output, symmetric_count); },
        [&] { return checksum(complex_output); });
}

void run_benchmarks_on_core(const Config& config, const topology::CoreClass& core_class,
                            const std::size_t class_index, const std::size_t class_count) {
    BenchmarkResults results;
    for (std::size_t profile_index = 0U; profile_index < kWorkloadProfiles.size(); ++profile_index) {
        run_benchmark_profile(config, kWorkloadProfiles[profile_index], profile_index, results);
    }
    results.print(core_class, class_index, class_count);
}

[[nodiscard]] std::string collect_members(const topology::Snapshot& snapshot, const std::string_view class_key) {
    std::vector<int> members;
    for (const auto& processor : snapshot.logical_processors) {
        const std::string_view key = processor.performance_class_key.empty() ? std::string_view{"default"}
                                                                             : processor.performance_class_key;
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

    fmt::print("{}\n| {:^{}} |\n{}\n", summary_frame, "CPU TOPOLOGY", summary_frame.size() - 4U,
               summary_border);
    print_summary_row("Model", result.snapshot.model_name);
    print_summary_row("Processors", fmt::format("{} logical / {} physical", result.snapshot.logical_processor_count,
                                                  result.snapshot.physical_core_count));
    print_summary_row("Packages / NUMA nodes", fmt::format("{} / {}", result.snapshot.package_count,
                                                             result.snapshot.numa_node_count));
    print_summary_row("Core types", std::to_string(core_classes.size()));
    print_summary_row("Working sets", "16 KiB / 512 KiB / 4 MiB / 64 MiB");
    print_summary_row("Repetitions", fmt::format("iterations={}, warmup={}", config.iterations,
                                                  config.warmup_iterations));
    fmt::print("{}\n\n", summary_border);

    constexpr std::array<std::size_t, 8U> core_widths{3U, 10U, 7U, 31U, 7U, 7U, 7U, 7U};
    const std::string core_border = table_border(core_widths);
    const std::string core_frame = outer_border(core_border);
    fmt::print("{}\n| {:^{}} |\n{}\n", core_frame, "CORE TYPES", core_frame.size() - 4U, core_border);
    fmt::print("| {:>3} | {:<10} | {:>7} | {:<31} | {:>7} | {:>7} | {:>7} | {:>7} |\n", "#", "type",
               "pin CPU", "logical CPU members", "L1d KiB", "L2 KiB", "L3 MiB", "GHz");
    fmt::print("{}\n", core_border);
    for (std::size_t index = 0U; index < core_classes.size(); ++index) {
        const auto& core_class = core_classes[index];
        const auto& processor = core_class.logical_processor;
        const auto cache_kib = [](const std::uint64_t bytes) {
            return bytes == 0U ? std::string{"n/a"} : fmt::format("{}", bytes / 1024U);
        };
        const std::string l1 = cache_kib(processor.l1_data_cache_bytes);
        const std::string l2 = cache_kib(processor.l2_cache_bytes);
        const std::string l3 = processor.l3_cache_bytes == 0U
                                   ? "n/a"
                                   : fmt::format("{:.1f}", static_cast<double>(processor.l3_cache_bytes) /
                                                               (1024.0 * 1024.0));
        const std::string frequency = processor.max_frequency_khz == 0U
                                          ? "n/a"
                                          : fmt::format("{:.2f}", static_cast<double>(processor.max_frequency_khz) /
                                                                    1'000'000.0);
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
        if (!affinity.is_pinned()) {
            throw std::runtime_error("failed to pin benchmark thread to logical CPU " +
                                     std::to_string(core_class.logical_processor.logical_processor_id));
        }
        run_benchmarks_on_core(config, core_class, index, core_classes.size());
    }
}

} // namespace

int main(const int argc, char** argv) {
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
