#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include <uni_simd.h>

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

constexpr std::array<uni_simd_backend_e, 7U> kCandidateBackends{
    UNI_SIMD_BACKEND_GENERIC,
    UNI_SIMD_BACKEND_X86_SSE2,
    UNI_SIMD_BACKEND_X86_AVX2,
    UNI_SIMD_BACKEND_X86_AVX2_FMA,
    UNI_SIMD_BACKEND_X86_AVX512,
    UNI_SIMD_BACKEND_AARCH64_NEON,
    UNI_SIMD_BACKEND_AUTOMATIC,
};
constexpr std::size_t kBackendSlots = UNI_SIMD_BACKEND_AARCH64_NEON + 1U;

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
    WorkloadProfile{"64 MiB", 64U * 1024U * 1024U, 2U, false},
};

enum class WorkloadShape : std::uint8_t {
    bytes,
    pack_bits,
    unpack_bits,
    complex_to_bytes,
    complex_to_float,
    dot,
    symmetric_dot,
    ifft,
    pfb,
};

enum class BackendPolicy : std::uint8_t {
    dispatched,
    runtime_only,
};

enum class PfbBenchmarkProfile : std::uint8_t {
    single_output,
    target_169,
    general_170,
    short_33,
};

enum Requirement : std::uint32_t {
    requirement_none = 0U,
    requirement_scale = 1U << 0U,
    requirement_normalization = 1U << 1U,
    requirement_rbw = 1U << 2U,
    requirement_taps = 1U << 3U,
    requirement_center_tap = 1U << 4U,
    requirement_transform_size = 1U << 5U,
    requirement_state = 1U << 6U,
};

struct KernelDescription {
    std::string_view name;
    std::string_view description;
    uni_simd_kernel_e kernel;
    WorkloadShape shape;
    std::uint32_t requirements;
    BackendPolicy backend_policy;
    std::size_t configuration = 0U;
    bool all_profiles = true;
    PfbBenchmarkProfile pfb_profile = PfbBenchmarkProfile::single_output;
};

constexpr std::array kKernelDescriptions{
    KernelDescription{"copy_u8", "byte copy with memmove semantics", UNI_SIMD_KERNEL_COPY_U8,
                      WorkloadShape::bytes, requirement_none, BackendPolicy::runtime_only},
    KernelDescription{"invert_lsb_u8", "invert the least significant bit", UNI_SIMD_KERNEL_INVERT_LSB_U8,
                      WorkloadShape::bytes, requirement_none, BackendPolicy::dispatched},
    KernelDescription{"invert_u8", "invert all bits", UNI_SIMD_KERNEL_INVERT_U8,
                      WorkloadShape::bytes, requirement_none, BackendPolicy::dispatched},
    KernelDescription{"pack_bits_lsb_u8", "pack byte-valued bits, LSB first", UNI_SIMD_KERNEL_PACK_BITS_LSB_U8,
                      WorkloadShape::pack_bits, requirement_none, BackendPolicy::dispatched},
    KernelDescription{"pack_bits_msb_u8", "pack byte-valued bits, MSB first", UNI_SIMD_KERNEL_PACK_BITS_MSB_U8,
                      WorkloadShape::pack_bits, requirement_none, BackendPolicy::dispatched},
    KernelDescription{"unpack_bits_lsb_u8", "unpack bits, LSB first", UNI_SIMD_KERNEL_UNPACK_BITS_LSB_U8,
                      WorkloadShape::unpack_bits, requirement_none, BackendPolicy::dispatched},
    KernelDescription{"unpack_bits_msb_u8", "unpack bits, MSB first", UNI_SIMD_KERNEL_UNPACK_BITS_MSB_U8,
                      WorkloadShape::unpack_bits, requirement_none, BackendPolicy::dispatched},
    KernelDescription{"quantize_cf32_u8", "quantize interleaved complex floats", UNI_SIMD_KERNEL_QUANTIZE_CF32_U8,
                      WorkloadShape::complex_to_bytes, requirement_scale, BackendPolicy::dispatched},
    KernelDescription{"magnitude_squared_cf32_f32", "compute normalized squared magnitude",
                      UNI_SIMD_KERNEL_MAGNITUDE_SQUARED_CF32_F32, WorkloadShape::complex_to_float,
                      requirement_normalization, BackendPolicy::dispatched},
    KernelDescription{"power_spectral_density_cf32_f32", "compute normalized power spectral density",
                      UNI_SIMD_KERNEL_POWER_SPECTRAL_DENSITY_CF32_F32, WorkloadShape::complex_to_float,
                      requirement_normalization | requirement_rbw, BackendPolicy::dispatched},
    KernelDescription{"dot_cf32_f32", "complex/real dot product", UNI_SIMD_KERNEL_DOT_CF32_F32,
                      WorkloadShape::dot, requirement_taps, BackendPolicy::dispatched},
    KernelDescription{"dot_symmetric_cf32_f32", "symmetric complex/real dot product",
                      UNI_SIMD_KERNEL_DOT_SYMMETRIC_CF32_F32, WorkloadShape::symmetric_dot,
                      requirement_taps | requirement_center_tap, BackendPolicy::dispatched},
    KernelDescription{"ifft_cf32_4", "four-point split-complex IFFT", UNI_SIMD_KERNEL_IFFT_SPLIT_CF32,
                      WorkloadShape::ifft, requirement_transform_size, BackendPolicy::dispatched, 4U, false},
    KernelDescription{"ifft_cf32_8", "eight-point split-complex IFFT", UNI_SIMD_KERNEL_IFFT_SPLIT_CF32,
                      WorkloadShape::ifft, requirement_transform_size, BackendPolicy::dispatched, 8U, false},
    KernelDescription{"ifft_cf32_16", "sixteen-point split-complex IFFT", UNI_SIMD_KERNEL_IFFT_SPLIT_CF32,
                      WorkloadShape::ifft, requirement_transform_size, BackendPolicy::dispatched, 16U, false},
    KernelDescription{"ifft_cf32_32", "thirty-two-point split-complex IFFT", UNI_SIMD_KERNEL_IFFT_SPLIT_CF32,
                      WorkloadShape::ifft, requirement_transform_size, BackendPolicy::dispatched, 32U, false},
    KernelDescription{"pfb_channelizer_cf32_4", "four-bin streaming PFB channelizer",
                      UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32, WorkloadShape::pfb,
                      requirement_transform_size | requirement_taps | requirement_state,
                      BackendPolicy::dispatched, 4U},
    KernelDescription{"pfb_cf32_8_target169", "eight-bin target profile: 169 taps, four half-grid outputs",
                      UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32, WorkloadShape::pfb,
                      requirement_transform_size | requirement_taps | requirement_state,
                      BackendPolicy::dispatched, 8U, true, PfbBenchmarkProfile::target_169},
    KernelDescription{"pfb_cf32_8_general170", "eight-bin general path: 170 taps, four half-grid outputs",
                      UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32, WorkloadShape::pfb,
                      requirement_transform_size | requirement_taps | requirement_state,
                      BackendPolicy::dispatched, 8U, true, PfbBenchmarkProfile::general_170},
    KernelDescription{"pfb_cf32_8_short33", "eight-bin short filter: 33 taps, one integer-grid output",
                      UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32, WorkloadShape::pfb,
                      requirement_transform_size | requirement_taps | requirement_state,
                      BackendPolicy::dispatched, 8U, true, PfbBenchmarkProfile::short_33},
    KernelDescription{"pfb_channelizer_cf32_16", "sixteen-bin streaming PFB channelizer",
                      UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32, WorkloadShape::pfb,
                      requirement_transform_size | requirement_taps | requirement_state,
                      BackendPolicy::dispatched, 16U},
    KernelDescription{"pfb_channelizer_cf32_32", "thirty-two-bin streaming PFB channelizer",
                      UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32, WorkloadShape::pfb,
                      requirement_transform_size | requirement_taps | requirement_state,
                      BackendPolicy::dispatched, 32U},
};

constexpr std::size_t kBufferAlignment = 128U;
constexpr std::size_t kProfileColumnWidth = 29U;
constexpr auto kCalibrationDuration = std::chrono::microseconds{250};
volatile double checksum_sink = 0.0;

struct Config {
    std::size_t iterations = 3U;
    std::size_t warmup = 1U;
    std::size_t sample_milliseconds = 1U;
    std::string_view preset = "quick";
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

class SimdRuntime final {
public:
    SimdRuntime() {
        if (uni_simd_initialize() != UNI_SIMD_RESULT_SUCCESS) {
            throw std::runtime_error("failed to initialize Uni.SIMD");
        }
    }

    ~SimdRuntime() {
        (void)uni_simd_finalize();
    }

    SimdRuntime(const SimdRuntime&) = delete;
    SimdRuntime& operator=(const SimdRuntime&) = delete;
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
    fmt::print("Usage: {} [--thorough] [--iterations N] [--warmup N] [--sample-ms N]\n", program);
}

[[nodiscard]] ParseResult parse_arguments(const int argc, char** argv, Config& config) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        if (argument == "--help" || argument == "-h") {
            return ParseResult::help;
        }
        if (argument == "--thorough") {
            config.iterations = 100U;
            config.warmup = 10U;
            config.sample_milliseconds = 10U;
            config.preset = "thorough";
            continue;
        }
        if (argument != "--iterations" && argument != "--samples" && argument != "--warmup" &&
            argument != "--sample-ms") {
            std::cerr << "Unknown argument: " << argument << '\n';
            return ParseResult::error;
        }
        if (++index == argc) {
            std::cerr << "Missing value for " << argument << '\n';
            return ParseResult::error;
        }
        const auto value = parse_size(argv[index]);
        if (!value) {
            std::cerr << "Invalid positive integer for " << argument << ": " << argv[index] << '\n';
            return ParseResult::error;
        }
        if (argument == "--iterations" || argument == "--samples") {
            config.iterations = *value;
        } else if (argument == "--warmup") {
            config.warmup = *value;
        } else {
            config.sample_milliseconds = *value;
        }
        config.preset = "custom";
    }
    return ParseResult::run;
}

[[nodiscard]] std::string_view backend_name(const uni_simd_backend_e backend) noexcept {
    switch (backend) {
    case UNI_SIMD_BACKEND_AUTOMATIC:
        return "automatic";
    case UNI_SIMD_BACKEND_GENERIC:
        return "generic";
    case UNI_SIMD_BACKEND_X86_SSE2:
        return "sse2";
    case UNI_SIMD_BACKEND_X86_AVX2:
        return "avx2";
    case UNI_SIMD_BACKEND_X86_AVX2_FMA:
        return "avx2-fma";
    case UNI_SIMD_BACKEND_X86_AVX512:
        return "avx512";
    case UNI_SIMD_BACKEND_AARCH64_NEON:
        return "neon";
    default:
        return "unknown";
    }
}

[[nodiscard]] double median(std::vector<double> values) {
    if (values.empty()) {
        throw std::runtime_error("benchmark produced no samples");
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2U;
    return values.size() % 2U != 0U ? values[middle] : (values[middle - 1U] + values[middle]) * 0.5;
}

template <typename Operation, typename Checksum>
[[nodiscard]] Statistics measure(const Config& config, const WorkloadProfile& profile,
                                 const std::size_t item_count, const double bytes_per_iteration,
                                 Operation&& operation, Checksum&& checksum) {
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
            if (operation(active_lane) != UNI_SIMD_RESULT_SUCCESS) {
                throw std::runtime_error("SIMD operation failed while measuring");
            }
        }
    };

    std::size_t calibration_repetitions = 1U;
    double calibration_seconds = 0.0;
    for (;;) {
        const std::size_t lane = take_lane();
        if (profile.prewarm_samples && operation(lane) != UNI_SIMD_RESULT_SUCCESS) {
            throw std::runtime_error("SIMD operation failed while calibrating");
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

    for (std::size_t iteration = 0U; iteration < config.warmup; ++iteration) {
        run_batch(take_lane(), batch_repetitions);
    }

    std::vector<double> nanoseconds_per_item;
    nanoseconds_per_item.reserve(config.iterations);
    std::size_t last_lane = 0U;
    for (std::size_t iteration = 0U; iteration < config.iterations; ++iteration) {
        last_lane = take_lane();
        if (profile.prewarm_samples && operation(last_lane) != UNI_SIMD_RESULT_SUCCESS) {
            throw std::runtime_error("SIMD operation failed while prewarming");
        }
        const auto begin = Clock::now();
        run_batch(last_lane, batch_repetitions);
        const double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
        if (seconds <= 0.0) {
            throw std::runtime_error("benchmark timer resolution is insufficient");
        }
        nanoseconds_per_item.push_back(
            seconds * 1.0e9 / (static_cast<double>(item_count) * static_cast<double>(batch_repetitions)));
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

template <typename Prepare, typename Operation, typename Checksum>
[[nodiscard]] Statistics measure_prepared(const Config& config, const WorkloadProfile& profile,
                                          const std::size_t item_count, const double bytes_per_iteration,
                                          Prepare&& prepare, Operation&& operation, Checksum&& checksum) {
    std::size_t next_lane = 0U;
    const auto take_lane = [&] {
        const std::size_t lane = next_lane;
        next_lane = (next_lane + 1U) % profile.address_lanes;
        return lane;
    };
    const auto run_once = [&](const std::size_t lane) {
        prepare(lane);
        const auto begin = Clock::now();
        if (operation(lane) != UNI_SIMD_RESULT_SUCCESS) {
            throw std::runtime_error("SIMD operation failed while measuring prepared input");
        }
        return std::chrono::duration<double>(Clock::now() - begin).count();
    };

    const double calibration_seconds = run_once(take_lane());
    if (calibration_seconds <= 0.0) {
        throw std::runtime_error("benchmark timer resolution is insufficient");
    }
    const double target_seconds = static_cast<double>(config.sample_milliseconds) / 1000.0;
    const double scaled_repetitions = std::ceil(target_seconds / calibration_seconds);
    if (scaled_repetitions > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("benchmark repetition count overflow");
    }
    const std::size_t repetitions = std::max<std::size_t>(1U, static_cast<std::size_t>(scaled_repetitions));

    for (std::size_t iteration = 0U; iteration < config.warmup; ++iteration) {
        for (std::size_t repetition = 0U; repetition < repetitions; ++repetition) {
            (void)run_once(take_lane());
        }
    }

    std::vector<double> nanoseconds_per_item;
    nanoseconds_per_item.reserve(config.iterations);
    std::size_t last_lane = 0U;
    for (std::size_t iteration = 0U; iteration < config.iterations; ++iteration) {
        double seconds = 0.0;
        for (std::size_t repetition = 0U; repetition < repetitions; ++repetition) {
            last_lane = take_lane();
            seconds += run_once(last_lane);
        }
        nanoseconds_per_item.push_back(
            seconds * 1.0e9 / (static_cast<double>(item_count) * static_cast<double>(repetitions)));
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

[[nodiscard]] uni_simd_param_t u32_parameter(const uni_simd_param_id_e id, const std::uint32_t value) noexcept {
    uni_simd_param_t parameter{.param_id = id, .param_type = UNI_SIMD_PARAM_TYPE_U32};
    parameter.value.u32 = value;
    return parameter;
}

[[nodiscard]] uni_simd_param_t size_parameter(const uni_simd_param_id_e id, const std::size_t value) noexcept {
    uni_simd_param_t parameter{.param_id = id, .param_type = UNI_SIMD_PARAM_TYPE_SIZE};
    parameter.value.size = value;
    return parameter;
}

[[nodiscard]] uni_simd_param_t float_parameter(const uni_simd_param_id_e id, const float value) noexcept {
    uni_simd_param_t parameter{.param_id = id, .param_type = UNI_SIMD_PARAM_TYPE_FLOAT32};
    parameter.value.f32 = value;
    return parameter;
}

[[nodiscard]] uni_simd_param_t pointer_parameter(const uni_simd_param_id_e id, const void* const value) noexcept {
    uni_simd_param_t parameter{.param_id = id, .param_type = UNI_SIMD_PARAM_TYPE_CONST_POINTER};
    parameter.value.const_pointer = value;
    return parameter;
}

[[nodiscard]] bool has_requirement(const KernelDescription& description, const Requirement requirement) noexcept {
    return (description.requirements & static_cast<std::uint32_t>(requirement)) != 0U;
}

struct Invocation {
    uni_simd_const_buffer_t input{};
    uni_simd_buffer_t output{};
    std::array<uni_simd_param_t, 8U> parameters{};
    std::size_t parameter_count = 0U;
};

struct StateDeleter {
    void operator()(uni_simd_state_t* const state) const noexcept {
        uni_simd_state_free(state);
    }
};

using StatePtr = std::unique_ptr<uni_simd_state_t, StateDeleter>;

class Workload final {
public:
    Workload(const KernelDescription& description, const WorkloadProfile& profile)
        : description_(description), profile_(profile) {
        prepare_shape();
        prepare_invocations();
        require_success(configure(UNI_SIMD_BACKEND_GENERIC), "generic configuration");
        require_success(execute(0U), "generic reference");
        capture_reference();
    }

    ~Workload() {
        free_states();
    }

    [[nodiscard]] uni_simd_result_e configure(const uni_simd_backend_e backend) {
        free_states();
        if (description_.shape == WorkloadShape::ifft) {
            fill_ifft_inputs();
        }
        for (std::size_t lane = 0U; lane < invocations_.size(); ++lane) {
            auto& invocation = invocations_[lane];
            std::size_t count = 0U;
            if (description_.shape == WorkloadShape::pfb) {
                invocation.parameters[count++] =
                    u32_parameter(UNI_SIMD_PARAM_RESOLVED_BACKEND, UNI_SIMD_BACKEND_AUTOMATIC);
                invocation.parameters[count++] = size_parameter(UNI_SIMD_PARAM_OUTPUT_COUNT, 0U);
                invocation.parameter_count = count;
                continue;
            }
            invocation.parameters[count++] = u32_parameter(UNI_SIMD_PARAM_BACKEND, backend);
            invocation.parameters[count++] = u32_parameter(UNI_SIMD_PARAM_RESOLVED_BACKEND, UNI_SIMD_BACKEND_AUTOMATIC);
            if (has_requirement(description_, requirement_scale)) {
                invocation.parameters[count++] = float_parameter(UNI_SIMD_PARAM_SCALE, -7.0f);
            }
            if (has_requirement(description_, requirement_normalization)) {
                invocation.parameters[count++] = float_parameter(UNI_SIMD_PARAM_NORMALIZATION_FACTOR, 3.0f);
            }
            if (has_requirement(description_, requirement_rbw)) {
                invocation.parameters[count++] = float_parameter(UNI_SIMD_PARAM_RBW_HZ, 2.0f);
            }
            if (has_requirement(description_, requirement_taps) && description_.shape != WorkloadShape::pfb) {
                invocation.parameters[count++] = pointer_parameter(UNI_SIMD_PARAM_TAPS, taps_[lane].data());
                invocation.parameters[count++] = size_parameter(UNI_SIMD_PARAM_TAP_COUNT, taps_[lane].size());
            }
            if (has_requirement(description_, requirement_center_tap)) {
                invocation.parameters[count++] = float_parameter(UNI_SIMD_PARAM_CENTER_TAP, 0.25f);
            }
            invocation.parameter_count = count;
        }
        if (description_.shape != WorkloadShape::pfb) {
            return UNI_SIMD_RESULT_SUCCESS;
        }
        for (std::size_t lane = 0U; lane < invocations_.size(); ++lane) {
            std::array<uni_simd_param_t, 9U> creation{
                u32_parameter(UNI_SIMD_PARAM_BACKEND, backend),
                u32_parameter(UNI_SIMD_PARAM_RESOLVED_BACKEND, UNI_SIMD_BACKEND_AUTOMATIC),
                size_parameter(UNI_SIMD_PARAM_BIN_COUNT, description_.configuration),
                size_parameter(UNI_SIMD_PARAM_DECIMATION, description_.configuration / 2U),
                u32_parameter(UNI_SIMD_PARAM_GRID_OFFSET,
                              description_.pfb_profile == PfbBenchmarkProfile::target_169 ||
                                      description_.pfb_profile == PfbBenchmarkProfile::general_170
                                  ? UNI_SIMD_PFB_HALF_BINS
                                  : UNI_SIMD_PFB_INTEGER_BINS),
                pointer_parameter(UNI_SIMD_PARAM_TAPS, taps_[lane].data()),
                size_parameter(UNI_SIMD_PARAM_TAP_COUNT, taps_[lane].size()),
                pointer_parameter(UNI_SIMD_PARAM_LOGICAL_BINS,
                                  description_.pfb_profile == PfbBenchmarkProfile::target_169 ||
                                          description_.pfb_profile == PfbBenchmarkProfile::general_170
                                      ? pfb_logical_bins_.data()
                                      : description_.pfb_profile == PfbBenchmarkProfile::short_33
                                            ? &pfb_three_bin_
                                            : &pfb_zero_bin_),
                size_parameter(UNI_SIMD_PARAM_LOGICAL_BIN_COUNT, pfb_output_count()),
            };
            uni_simd_state_t* state = nullptr;
            const uni_simd_result_e result = uni_simd_execute(
                description_.kernel, nullptr, nullptr, creation.data(), creation.size(), &state);
            if (result != UNI_SIMD_RESULT_SUCCESS) {
                uni_simd_state_free(state);
                free_states();
                return result;
            }
            states_[lane].reset(state);
            invocations_[lane].parameters[0].value.u32 = creation[1].value.u32;
        }
        return UNI_SIMD_RESULT_SUCCESS;
    }

    [[nodiscard]] uni_simd_result_e execute(const std::size_t lane) {
        auto& invocation = invocations_.at(lane);
        if (description_.shape == WorkloadShape::ifft) {
            uni_simd_split_cf32_t values{
                .real = ifft_real_[lane].data(),
                .imag = ifft_imag_[lane].data(),
                .descriptor_size = UNI_SIMD_SPLIT_CF32_DESCRIPTOR_SIZE,
                .transform_size = description_.configuration,
                .transform_count = item_count_ / description_.configuration,
                .stride = description_.configuration,
            };
            return uni_simd_execute(description_.kernel, nullptr, &values,
                                    invocation.parameters.data(), invocation.parameter_count, nullptr);
        }
        if (description_.shape == WorkloadShape::pfb) {
            uni_simd_state_t* state = states_[lane].get();
            return uni_simd_execute(description_.kernel, &invocation.input, &pfb_output_arrays_[lane],
                                    invocation.parameters.data(), invocation.parameter_count, &state);
        }
        return uni_simd_execute(description_.kernel, &invocation.input, &invocation.output,
                                invocation.parameters.data(), invocation.parameter_count, nullptr);
    }

    [[nodiscard]] uni_simd_backend_e resolved_backend() const noexcept {
        const std::size_t parameter_index = description_.shape == WorkloadShape::pfb ? 0U : 1U;
        return invocations_.front().parameters[parameter_index].value.u32;
    }

    void prepare_measurement(const std::size_t lane) {
        if (description_.shape == WorkloadShape::ifft) {
            fill_ifft_input(lane);
        }
    }

    [[nodiscard]] bool needs_prepared_measurement() const noexcept {
        return description_.shape == WorkloadShape::ifft;
    }

    void validate(const std::size_t lane) const {
        if (!u8_reference_.empty()) {
            if (u8_outputs_.at(lane) != u8_reference_) {
                throw std::runtime_error(error_message("result differs from generic reference"));
            }
            return;
        }
        std::span<const float> actual;
        AlignedBuffer<float> ifft_actual;
        if (description_.shape == WorkloadShape::ifft) {
            ifft_actual.reserve(item_count_ * 2U);
            ifft_actual.insert(ifft_actual.end(), ifft_real_[lane].begin(), ifft_real_[lane].end());
            ifft_actual.insert(ifft_actual.end(), ifft_imag_[lane].begin(), ifft_imag_[lane].end());
            actual = ifft_actual;
        } else {
            actual = f32_outputs_.at(lane);
        }
        if (actual.size() != f32_reference_.size()) {
            throw std::runtime_error(error_message("result size differs from generic reference"));
        }
        const float accumulated_tolerance =
            description_.shape == WorkloadShape::dot || description_.shape == WorkloadShape::symmetric_dot ||
                    description_.shape == WorkloadShape::pfb
                ? 1.0e-4f + 8.0f * std::numeric_limits<float>::epsilon() *
                                 std::sqrt(static_cast<float>(description_.shape == WorkloadShape::pfb
                                                                  ? taps_.front().size()
                                                                  : item_count_))
                : 1.0e-5f;
        for (std::size_t index = 0U; index < actual.size(); ++index) {
            const float tolerance = accumulated_tolerance * std::max(1.0f, std::abs(f32_reference_[index]));
            if (!std::isfinite(actual[index]) || !std::isfinite(f32_reference_[index]) ||
                std::abs(actual[index] - f32_reference_[index]) > tolerance) {
                throw std::runtime_error(error_message("result differs from generic reference"));
            }
        }
    }

    [[nodiscard]] double checksum(const std::size_t lane) const noexcept {
        double result = 0.0;
        if (!u8_outputs_.empty()) {
            for (const std::uint8_t value : u8_outputs_[lane]) {
                result += value;
            }
        } else if (description_.shape == WorkloadShape::ifft) {
            for (const float value : ifft_real_[lane]) {
                result += value;
            }
            for (const float value : ifft_imag_[lane]) {
                result += value;
            }
        } else {
            for (const float value : f32_outputs_[lane]) {
                result += value;
            }
        }
        return result;
    }

    [[nodiscard]] std::size_t item_count() const noexcept { return item_count_; }
    [[nodiscard]] double bytes_per_iteration() const noexcept { return bytes_per_iteration_; }

private:
    void prepare_shape() {
        const std::size_t lanes = profile_.address_lanes;
        switch (description_.shape) {
        case WorkloadShape::bytes:
            item_count_ = std::max<std::size_t>(64U, profile_.working_set_bytes / 2U);
            u8_inputs_ = make_lanes<std::uint8_t>(lanes, item_count_);
            u8_outputs_ = make_lanes<std::uint8_t>(lanes, item_count_);
            break;
        case WorkloadShape::pack_bits: {
            const std::size_t requested = std::max<std::size_t>(64U, profile_.working_set_bytes * 8U / 9U);
            item_count_ = requested - requested % 8U;
            u8_inputs_ = make_lanes<std::uint8_t>(lanes, item_count_);
            u8_outputs_ = make_lanes<std::uint8_t>(lanes, item_count_ / 8U);
            break;
        }
        case WorkloadShape::unpack_bits: {
            const std::size_t requested = std::max<std::size_t>(64U, profile_.working_set_bytes * 8U / 9U);
            item_count_ = requested - requested % 8U;
            u8_inputs_ = make_lanes<std::uint8_t>(lanes, item_count_ / 8U);
            u8_outputs_ = make_lanes<std::uint8_t>(lanes, item_count_);
            break;
        }
        case WorkloadShape::complex_to_bytes:
            item_count_ = std::max<std::size_t>(8U, profile_.working_set_bytes / 10U);
            f32_inputs_ = make_lanes<float>(lanes, item_count_ * 2U);
            u8_outputs_ = make_lanes<std::uint8_t>(lanes, item_count_ * 2U);
            break;
        case WorkloadShape::complex_to_float:
            item_count_ = std::max<std::size_t>(8U, profile_.working_set_bytes / 12U);
            f32_inputs_ = make_lanes<float>(lanes, item_count_ * 2U);
            f32_outputs_ = make_lanes<float>(lanes, item_count_);
            break;
        case WorkloadShape::dot:
            item_count_ = std::max<std::size_t>(8U, profile_.working_set_bytes / 12U);
            f32_inputs_ = make_lanes<float>(lanes, item_count_ * 2U);
            f32_outputs_ = make_lanes<float>(lanes, 2U);
            taps_ = make_lanes<float>(lanes, item_count_);
            break;
        case WorkloadShape::symmetric_dot:
            item_count_ = std::max<std::size_t>(9U, profile_.working_set_bytes / 10U);
            if (item_count_ % 2U == 0U) {
                --item_count_;
            }
            f32_inputs_ = make_lanes<float>(lanes, item_count_ * 2U);
            f32_outputs_ = make_lanes<float>(lanes, 2U);
            taps_ = make_lanes<float>(lanes, (item_count_ - 1U) / 2U);
            break;
        case WorkloadShape::ifft:
            item_count_ = std::max<std::size_t>(
                description_.configuration,
                profile_.working_set_bytes / (2U * sizeof(float)));
            item_count_ -= item_count_ % description_.configuration;
            ifft_real_ = make_lanes<float>(lanes, item_count_);
            ifft_imag_ = make_lanes<float>(lanes, item_count_);
            break;
        case WorkloadShape::pfb: {
            const std::size_t decimation = description_.configuration / 2U;
            const std::size_t output_count = pfb_output_count();
            item_count_ = std::max<std::size_t>(
                decimation * 8U,
                profile_.working_set_bytes * decimation /
                    (2U * sizeof(float) * (decimation + output_count)));
            item_count_ -= item_count_ % decimation;
            f32_inputs_ = make_lanes<float>(lanes, item_count_ * 2U);
            f32_outputs_ = make_lanes<float>(lanes, item_count_ / decimation * 2U * output_count);
            const std::size_t tap_count = description_.pfb_profile == PfbBenchmarkProfile::target_169
                                              ? 169U
                                          : description_.pfb_profile == PfbBenchmarkProfile::general_170
                                              ? 170U
                                          : description_.pfb_profile == PfbBenchmarkProfile::short_33
                                              ? 33U
                                              : description_.configuration * 8U;
            taps_ = make_lanes<float>(lanes, tap_count);
            states_.resize(lanes);
            pfb_output_buffers_.resize(lanes);
            for (auto& buffers : pfb_output_buffers_) {
                buffers.resize(output_count);
            }
            pfb_output_arrays_.resize(lanes);
            break;
        }
        }

        for (auto& input : u8_inputs_) {
            for (std::size_t index = 0U; index < input.size(); ++index) {
                input[index] = description_.shape == WorkloadShape::pack_bits
                                   ? static_cast<std::uint8_t>((index * 5U + 1U) & 1U)
                                   : static_cast<std::uint8_t>((index * 37U + 11U) & 0xffU);
            }
        }
        for (auto& input : f32_inputs_) {
            for (std::size_t sample = 0U; sample < input.size() / 2U; ++sample) {
                input[2U * sample] = static_cast<float>(sample % 127U) / 63.5f - 1.0f;
                input[2U * sample + 1U] = static_cast<float>(sample % 61U) / 30.5f - 1.0f;
            }
        }
        for (auto& lane_taps : taps_) {
            for (std::size_t index = 0U; index < lane_taps.size(); ++index) {
                lane_taps[index] = static_cast<float>(static_cast<int>(index % 31U) - 15) / 31.0f;
            }
        }

        if (description_.shape == WorkloadShape::ifft) {
            fill_ifft_inputs();
            bytes_per_iteration_ = static_cast<double>(item_count_ * 4U * sizeof(float));
            return;
        }

        const double input_bytes = !u8_inputs_.empty()
                                       ? static_cast<double>(u8_inputs_.front().size())
                                       : static_cast<double>(f32_inputs_.front().size() * sizeof(float));
        const double output_bytes = !u8_outputs_.empty()
                                        ? static_cast<double>(u8_outputs_.front().size())
                                        : static_cast<double>(f32_outputs_.front().size() * sizeof(float));
        const bool taps_are_streamed = description_.shape == WorkloadShape::dot ||
                                       description_.shape == WorkloadShape::symmetric_dot;
        const double tap_bytes = !taps_are_streamed || taps_.empty()
                                     ? 0.0
                                     : static_cast<double>(taps_.front().size() * sizeof(float));
        bytes_per_iteration_ = input_bytes + output_bytes + tap_bytes;
    }

    void prepare_invocations() {
        invocations_.resize(profile_.address_lanes);
        for (std::size_t lane = 0U; lane < invocations_.size(); ++lane) {
            auto& invocation = invocations_[lane];
            if (description_.shape == WorkloadShape::ifft) {
                continue;
            }
            if (!u8_inputs_.empty()) {
                invocation.input = {u8_inputs_[lane].data(), u8_inputs_[lane].size()};
            } else {
                invocation.input = {f32_inputs_[lane].data(), item_count_};
            }
            if (!u8_outputs_.empty()) {
                invocation.output = {u8_outputs_[lane].data(), u8_outputs_[lane].size()};
            } else {
                const std::size_t output_count =
                    description_.shape == WorkloadShape::dot || description_.shape == WorkloadShape::symmetric_dot
                        ? 1U
                        : f32_outputs_[lane].size();
                invocation.output = {f32_outputs_[lane].data(), output_count};
            }
            if (description_.shape == WorkloadShape::pfb) {
                const std::size_t samples_per_output =
                    item_count_ / (description_.configuration / 2U);
                for (std::size_t output_index = 0U; output_index < pfb_output_count(); ++output_index) {
                    pfb_output_buffers_[lane][output_index] = {
                        .data = f32_outputs_[lane].data() + output_index * samples_per_output * 2U,
                        .count = samples_per_output,
                    };
                }
                pfb_output_arrays_[lane] = {
                    .buffers = pfb_output_buffers_[lane].data(),
                    .count = pfb_output_buffers_[lane].size(),
                };
            }
        }
    }

    void capture_reference() {
        if (!u8_outputs_.empty()) {
            u8_reference_ = u8_outputs_.front();
        } else if (description_.shape == WorkloadShape::ifft) {
            f32_reference_.reserve(item_count_ * 2U);
            f32_reference_.insert(f32_reference_.end(), ifft_real_.front().begin(), ifft_real_.front().end());
            f32_reference_.insert(f32_reference_.end(), ifft_imag_.front().begin(), ifft_imag_.front().end());
        } else {
            f32_reference_ = f32_outputs_.front();
        }
    }

    void fill_ifft_inputs() {
        for (std::size_t lane = 0U; lane < ifft_real_.size(); ++lane) {
            fill_ifft_input(lane);
        }
    }

    void fill_ifft_input(const std::size_t lane) {
        for (std::size_t index = 0U; index < item_count_; ++index) {
            ifft_real_[lane][index] = static_cast<float>((index * 7U + 3U) % 17U) / 17.0f - 0.5f;
            ifft_imag_[lane][index] = static_cast<float>((index * 5U + 1U) % 13U) / 13.0f - 0.5f;
        }
    }

    void free_states() noexcept {
        for (auto& state : states_) {
            state.reset();
        }
    }

    [[nodiscard]] std::size_t pfb_output_count() const noexcept {
        return description_.pfb_profile == PfbBenchmarkProfile::target_169 ||
                       description_.pfb_profile == PfbBenchmarkProfile::general_170
                   ? pfb_logical_bins_.size()
                   : 1U;
    }

    void require_success(const uni_simd_result_e result, const std::string_view phase) const {
        if (result != UNI_SIMD_RESULT_SUCCESS) {
            throw std::runtime_error(error_message(fmt::format("{} failed with result {}", phase, result)));
        }
    }

    [[nodiscard]] std::string error_message(const std::string_view message) const {
        return fmt::format("{} ({}): {}", description_.name, description_.description, message);
    }

    const KernelDescription& description_;
    const WorkloadProfile& profile_;
    std::size_t item_count_ = 0U;
    double bytes_per_iteration_ = 0.0;
    BufferLanes<std::uint8_t> u8_inputs_;
    BufferLanes<std::uint8_t> u8_outputs_;
    BufferLanes<float> f32_inputs_;
    BufferLanes<float> f32_outputs_;
    BufferLanes<float> taps_;
    BufferLanes<float> ifft_real_;
    BufferLanes<float> ifft_imag_;
    AlignedBuffer<std::uint8_t> u8_reference_;
    AlignedBuffer<float> f32_reference_;
    std::vector<Invocation> invocations_;
    std::vector<StatePtr> states_;
    std::vector<std::vector<uni_simd_buffer_t>> pfb_output_buffers_;
    std::vector<uni_simd_buffer_array_t> pfb_output_arrays_;
    std::array<std::int32_t, 4U> pfb_logical_bins_{-2, -1, 0, 1};
    std::int32_t pfb_zero_bin_ = 0;
    std::int32_t pfb_three_bin_ = 3;
};

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
            result[0] = 34U;
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
        fmt::print("| {:<34} | {:<8} |", "kernel", "backend");
        for (const auto& profile : kWorkloadProfiles) {
            fmt::print(" {:^{}} |", profile.label, kProfileColumnWidth);
        }
        fmt::print("\n| {:<34} | {:<8} |", "", "");
        const std::string metric_header =
            fmt::format("{:>6} {:>5} {:>5} {:>10}", "ns/i", "GiB/s", "MAD%", "vs generic");
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
            fmt::print("| {:<34} | {:<8} |", row.name, row.backend);
            for (const auto& profile : row.profiles) {
                fmt::print(" {:>{}} |", format_profile(profile), kProfileColumnWidth);
            }
            fmt::print("\n");
        }
        fmt::print("{}\n", border);
    }

private:
    [[nodiscard]] static std::string format_profile(const std::optional<ProfileMeasurement>& measurement) {
        if (!measurement) {
            return "-";
        }
        const std::string speedup = measurement->speedup ? fmt::format("{:.2f}x", *measurement->speedup) : "-";
        const std::string mad = fmt::format("{:.1f}%", measurement->statistics.relative_mad_percent);
        return fmt::format("{:>6.3f} {:>5.1f} {:>5} {:>10}", measurement->statistics.nanoseconds_per_item,
                           measurement->statistics.gibibytes_per_second, mad, speedup);
    }

    std::vector<ResultRow> rows_;
};

[[nodiscard]] Statistics measure_workload(const Config& config, const WorkloadProfile& profile,
                                          Workload& workload) {
    if (workload.needs_prepared_measurement()) {
        return measure_prepared(
            config, profile, workload.item_count(), workload.bytes_per_iteration(),
            [&](const std::size_t lane) { workload.prepare_measurement(lane); },
            [&](const std::size_t lane) { return workload.execute(lane); },
            [&](const std::size_t lane) { return workload.checksum(lane); });
    }
    return measure(
        config, profile, workload.item_count(), workload.bytes_per_iteration(),
        [&](const std::size_t lane) { return workload.execute(lane); },
        [&](const std::size_t lane) { return workload.checksum(lane); });
}

void run_kernel(const KernelDescription& description, const Config& config, const WorkloadProfile& profile,
                const std::size_t profile_index, BenchmarkResults& results) {
    Workload workload{description, profile};
    if (description.backend_policy == BackendPolicy::runtime_only) {
        if (workload.configure(UNI_SIMD_BACKEND_AUTOMATIC) != UNI_SIMD_RESULT_SUCCESS) {
            throw std::runtime_error(fmt::format("{} runtime configuration failed", description.name));
        }
        if (workload.execute(0U) != UNI_SIMD_RESULT_SUCCESS) {
            throw std::runtime_error(fmt::format("{} validation execution failed", description.name));
        }
        workload.validate(0U);
        const Statistics statistics = measure_workload(config, profile, workload);
        results.add(description.name, "runtime", profile_index, statistics, std::nullopt);
        return;
    }

    std::array<bool, kBackendSlots> measured{};
    std::optional<double> generic_nanoseconds;
    for (const uni_simd_backend_e requested : kCandidateBackends) {
        const uni_simd_result_e configuration_result = workload.configure(requested);
        if (configuration_result == UNI_SIMD_RESULT_UNSUPPORTED_BACKEND) {
            continue;
        }
        if (configuration_result != UNI_SIMD_RESULT_SUCCESS) {
            throw std::runtime_error(fmt::format("{} configuration failed for requested backend {} with result {}",
                                                 description.name, backend_name(requested), configuration_result));
        }
        const uni_simd_result_e result = workload.execute(0U);
        if (result == UNI_SIMD_RESULT_UNSUPPORTED_BACKEND) {
            continue;
        }
        if (result != UNI_SIMD_RESULT_SUCCESS) {
            throw std::runtime_error(fmt::format("{} failed for requested backend {} with result {}",
                                                 description.name, backend_name(requested), result));
        }
        const uni_simd_backend_e resolved = workload.resolved_backend();
        if (resolved >= measured.size()) {
            throw std::runtime_error(fmt::format("{} returned invalid backend {}", description.name, resolved));
        }
        if (measured[resolved]) {
            continue;
        }
        measured[resolved] = true;
        workload.validate(0U);
        const Statistics statistics = measure_workload(config, profile, workload);
        if (resolved == UNI_SIMD_BACKEND_GENERIC) {
            generic_nanoseconds = statistics.nanoseconds_per_item;
        }
        const std::optional<double> speedup = generic_nanoseconds
                                                  ? std::optional<double>{*generic_nanoseconds / statistics.nanoseconds_per_item}
                                                  : std::nullopt;
        results.add(description.name, backend_name(resolved), profile_index, statistics, speedup);
    }
}

void run_benchmark_profile(const Config& config, const WorkloadProfile& profile,
                           const std::size_t profile_index, BenchmarkResults& results) {
    for (const auto& description : kKernelDescriptions) {
        if (!description.all_profiles && profile_index != 0U) {
            continue;
        }
        run_kernel(description, config, profile, profile_index, results);
    }
}

void run_benchmarks_on_core(const Config& config, const topology::CoreClass& core_class,
                            const topology::ThreadAffinityStatus affinity_status,
                            const std::size_t class_index, const std::size_t class_count) {
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
    print_summary_row("Packages / NUMA nodes", fmt::format("{} / {}", result.snapshot.package_count,
                                                            result.snapshot.numa_node_count));
    print_summary_row("Core types", std::to_string(core_classes.size()));
    print_summary_row("Kernels", fmt::format("{} table entries covering 14 public kernels", kKernelDescriptions.size()));
    print_summary_row("Working sets", std::move(working_sets));
    print_summary_row("Measurement", fmt::format("{}: iterations={}, warmup={}, target={} ms", config.preset,
                                                 config.iterations, config.warmup, config.sample_milliseconds));
    print_summary_row("Buffers", fmt::format("{}-byte aligned; address lanes={}", kBufferAlignment, address_lanes));
    print_summary_row("Cache policy", "rotate addresses; prewarm through 32 MiB; stream 64 MiB");
    print_summary_row("PFB working set", "caller payload only; kernels process interleaved buffers directly");
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
        const SimdRuntime runtime;
        run_benchmarks(config);
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
