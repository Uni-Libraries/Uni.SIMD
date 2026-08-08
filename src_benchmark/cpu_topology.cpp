#include "cpu_topology.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif
#endif

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

namespace uni::simd::benchmark::cpu_topology {
namespace {

constexpr int kUnknownId = -1;

[[nodiscard]] std::string_view trim(std::string_view value) noexcept {
    constexpr std::string_view whitespace = " \t\n\r";
    const auto first = value.find_first_not_of(whitespace);
    if (first == std::string_view::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(whitespace);
    return value.substr(first, last - first + 1U);
}

template <typename Value>
[[nodiscard]] std::optional<Value> parse_integer(std::string_view value) {
    value = trim(value);
    Value parsed{};
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (value.empty() || error != std::errc{} || end != value.data() + value.size()) {
        return std::nullopt;
    }
    return parsed;
}

[[nodiscard]] std::optional<std::string> read_first_line(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::string line;
    if (!input || !std::getline(input, line)) {
        return std::nullopt;
    }
    return line;
}

template <typename Value>
[[nodiscard]] std::optional<Value> read_integer(const std::filesystem::path& path) {
    const auto line = read_first_line(path);
    return line.has_value() ? parse_integer<Value>(*line) : std::nullopt;
}

[[nodiscard]] std::vector<int> parse_cpu_list(std::string_view text) {
    std::vector<int> cpus;
    std::size_t begin = 0U;
    while (begin <= text.size()) {
        const std::size_t comma = text.find(',', begin);
        const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
        const std::string_view token = trim(text.substr(begin, end - begin));
        const std::size_t dash = token.find('-');
        if (!token.empty() && dash == std::string_view::npos) {
            const auto cpu = parse_integer<int>(token);
            if (cpu.has_value() && *cpu >= 0) {
                cpus.push_back(*cpu);
            }
        } else if (!token.empty()) {
            const auto first = parse_integer<int>(token.substr(0U, dash));
            const auto last = parse_integer<int>(token.substr(dash + 1U));
            if (first.has_value() && last.has_value() && *first >= 0 && *last >= *first) {
                for (int cpu = *first; cpu <= *last; ++cpu) {
                    cpus.push_back(cpu);
                }
            }
        }
        if (comma == std::string_view::npos) {
            break;
        }
        begin = comma + 1U;
    }
    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    return cpus;
}

[[nodiscard]] std::vector<int> fallback_cpus() {
    const unsigned count = std::thread::hardware_concurrency();
    std::vector<int> cpus;
    cpus.reserve(count == 0U ? 1U : count);
    for (unsigned cpu = 0U; cpu < std::max(1U, count); ++cpu) {
        cpus.push_back(static_cast<int>(cpu));
    }
    return cpus;
}

void fill_aggregate_fields(Snapshot& snapshot) {
    snapshot.logical_processor_count = static_cast<std::uint32_t>(snapshot.logical_processors.size());
    std::set<std::pair<int, int>> physical_cores;
    std::set<int> packages;
    std::set<int> numa_nodes;
    bool all_have_core = !snapshot.logical_processors.empty();
    bool all_have_package = !snapshot.logical_processors.empty();

    for (const auto& processor : snapshot.logical_processors) {
        all_have_core = all_have_core && processor.core_id >= 0;
        all_have_package = all_have_package && processor.package_id >= 0;
        if (processor.core_id >= 0 && processor.package_id >= 0) {
            physical_cores.emplace(processor.package_id, processor.core_id);
        }
        if (processor.package_id >= 0) {
            packages.insert(processor.package_id);
        }
        if (processor.numa_node_id >= 0) {
            numa_nodes.insert(processor.numa_node_id);
        }
    }

    snapshot.has_core_mapping = all_have_core;
    snapshot.has_package_mapping = all_have_package;
    snapshot.has_numa_mapping = !numa_nodes.empty();
    snapshot.physical_core_count = static_cast<std::uint32_t>(
        physical_cores.empty() ? snapshot.logical_processors.size() : physical_cores.size());
    snapshot.package_count = static_cast<std::uint32_t>(
        packages.empty() ? (snapshot.logical_processors.empty() ? 0U : 1U) : packages.size());
    snapshot.numa_node_count = static_cast<std::uint32_t>(numa_nodes.size());
}

[[nodiscard]] Result unavailable_result(std::string message) {
    Result result;
    result.status = {.code = StatusCode::unavailable, .message = std::move(message)};
    return result;
}

[[nodiscard]] Result fallback_result(std::string message) {
    Result result;
    result.status = {.code = StatusCode::partial, .message = std::move(message)};
    for (const int cpu : fallback_cpus()) {
        result.snapshot.logical_processors.push_back({
            .logical_processor_id = cpu,
            .core_id = cpu,
            .package_id = 0,
            .group_id = static_cast<std::uint16_t>(cpu / 64),
            .group_index = static_cast<std::uint8_t>(cpu % 64),
            .performance_class_label = "cpu" + std::to_string(cpu),
        });
    }
    fill_aggregate_fields(result.snapshot);
    return result;
}

[[nodiscard]] std::string decimal_label(const double value, const int precision) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(precision) << value;
    return output.str();
}

[[nodiscard]] std::array<std::uint32_t, 4U> cpuid(const std::uint32_t leaf, const std::uint32_t subleaf) {
    std::array<std::uint32_t, 4U> output{};
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    std::array<int, 4U> registers{};
    __cpuidex(registers.data(), static_cast<int>(leaf), static_cast<int>(subleaf));
    for (std::size_t index = 0U; index < output.size(); ++index) {
        output[index] = static_cast<std::uint32_t>(registers[index]);
    }
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__i386__) || defined(__x86_64__))
    std::uint32_t a = 0U;
    std::uint32_t b = 0U;
    std::uint32_t c = 0U;
    std::uint32_t d = 0U;
    __cpuid_count(leaf, subleaf, a, b, c, d);
    output = {a, b, c, d};
#else
    (void)leaf;
    (void)subleaf;
#endif
    return output;
}

[[nodiscard]] std::string x86_brand() {
    if (cpuid(0x80000000U, 0U)[0] < 0x80000004U) {
        return {};
    }
    std::array<char, 49U> brand{};
    for (std::uint32_t index = 0U; index < 3U; ++index) {
        const auto registers = cpuid(0x80000002U + index, 0U);
        std::memcpy(brand.data() + index * 16U, registers.data(), 16U);
    }
    return std::string(trim(brand.data()));
}

#if defined(__linux__)

[[nodiscard]] std::filesystem::path cpu_path(const int cpu, const std::string_view suffix) {
    return std::filesystem::path("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + std::string(suffix));
}

[[nodiscard]] std::vector<int> allowed_cpus_linux() {
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (pthread_getaffinity_np(pthread_self(), sizeof(affinity), &affinity) != 0) {
        return {};
    }
    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(static_cast<unsigned>(cpu), &affinity)) {
            cpus.push_back(cpu);
        }
    }
    return cpus;
}

[[nodiscard]] std::vector<int> online_cpus_linux() {
    std::vector<int> online;
    if (const auto text = read_first_line("/sys/devices/system/cpu/online"); text.has_value()) {
        online = parse_cpu_list(*text);
    }
    const std::vector<int> allowed = allowed_cpus_linux();
    if (allowed.empty()) {
        return online.empty() ? fallback_cpus() : online;
    }
    if (online.empty()) {
        return allowed;
    }
    std::vector<int> intersection;
    std::set_intersection(online.begin(), online.end(), allowed.begin(), allowed.end(),
                          std::back_inserter(intersection));
    return intersection.empty() ? allowed : intersection;
}

[[nodiscard]] std::optional<std::uint64_t> parse_cache_size(std::string_view value) {
    value = trim(value);
    if (value.empty()) {
        return std::nullopt;
    }
    std::uint64_t multiplier = 1U;
    const char suffix = value.back();
    if (suffix == 'K' || suffix == 'k') {
        multiplier = 1024ULL;
        value.remove_suffix(1U);
    } else if (suffix == 'M' || suffix == 'm') {
        multiplier = 1024ULL * 1024ULL;
        value.remove_suffix(1U);
    } else if (suffix == 'G' || suffix == 'g') {
        multiplier = 1024ULL * 1024ULL * 1024ULL;
        value.remove_suffix(1U);
    }
    const auto size = parse_integer<std::uint64_t>(value);
    if (!size.has_value() || *size > std::numeric_limits<std::uint64_t>::max() / multiplier) {
        return std::nullopt;
    }
    return *size * multiplier;
}

[[nodiscard]] std::uint64_t read_cache_bytes_linux(const int cpu, const int wanted_level,
                                                    const bool data_only) {
    const auto cache_path = cpu_path(cpu, "/cache");
    std::error_code error;
    for (std::filesystem::directory_iterator iterator(cache_path, error), end; !error && iterator != end;
         iterator.increment(error)) {
        if (!iterator->is_directory(error)) {
            continue;
        }
        const auto level = read_integer<int>(iterator->path() / "level");
        const auto type = read_first_line(iterator->path() / "type");
        const auto size = read_first_line(iterator->path() / "size");
        if (!level.has_value() || *level != wanted_level || !type.has_value() || !size.has_value()) {
            continue;
        }
        const std::string_view cache_type = trim(*type);
        if (data_only ? cache_type != "Data" && cache_type != "Unified"
                      : cache_type != "Unified" && cache_type != "Data") {
            continue;
        }
        return parse_cache_size(*size).value_or(0U);
    }
    return 0U;
}

[[nodiscard]] std::optional<int> read_numa_node_linux(const int cpu) {
    const auto path = cpu_path(cpu, "");
    std::error_code error;
    for (std::filesystem::directory_iterator iterator(path, error), end; !error && iterator != end;
         iterator.increment(error)) {
        const std::string name = iterator->path().filename().string();
        if (name.starts_with("node")) {
            const auto node = parse_integer<int>(std::string_view{name}.substr(4U));
            if (node.has_value() && *node >= 0) {
                return node;
            }
        }
    }
    return std::nullopt;
}

[[nodiscard]] std::string x86_vendor() {
#if defined(__i386__) || defined(__x86_64__)
    const auto registers = cpuid(0U, 0U);
    std::array<char, 13U> vendor{};
    std::memcpy(vendor.data(), &registers[1], sizeof(std::uint32_t));
    std::memcpy(vendor.data() + 4U, &registers[3], sizeof(std::uint32_t));
    std::memcpy(vendor.data() + 8U, &registers[2], sizeof(std::uint32_t));
    return vendor.data();
#else
    return {};
#endif
}

[[nodiscard]] std::string cpu_model_linux() {
    std::ifstream input("/proc/cpuinfo");
    std::string line;
    while (std::getline(input, line)) {
        const std::string_view value = trim(line);
        if (!value.starts_with("model name")) {
            continue;
        }
        const std::size_t colon = value.find(':');
        if (colon != std::string_view::npos) {
            return std::string(trim(value.substr(colon + 1U)));
        }
    }
    return x86_brand();
}

struct ScopedCpuPinLinux {
    explicit ScopedCpuPinLinux(const int cpu) {
        CPU_ZERO(&previous);
        restore = pthread_getaffinity_np(pthread_self(), sizeof(previous), &previous) == 0;
        if (!restore || cpu < 0 || cpu >= CPU_SETSIZE) {
            return;
        }
        cpu_set_t target;
        CPU_ZERO(&target);
        CPU_SET(static_cast<unsigned>(cpu), &target);
        pinned = pthread_setaffinity_np(pthread_self(), sizeof(target), &target) == 0;
    }

    ~ScopedCpuPinLinux() {
        if (restore) {
            (void)pthread_setaffinity_np(pthread_self(), sizeof(previous), &previous);
        }
    }

    cpu_set_t previous{};
    bool restore = false;
    bool pinned = false;
};

[[nodiscard]] std::optional<std::uint32_t> intel_core_type(const int cpu) {
#if defined(__i386__) || defined(__x86_64__)
    ScopedCpuPinLinux affinity(cpu);
    if (!affinity.pinned || cpuid(0U, 0U)[0] < 0x1AU) {
        return std::nullopt;
    }
    const std::uint32_t core_type = (cpuid(0x1AU, 0U)[0] >> 24U) & 0xffU;
    return core_type == 0U ? std::nullopt : std::optional{core_type};
#else
    (void)cpu;
    return std::nullopt;
#endif
}

struct AmdFamilyModel {
    std::uint32_t family;
    std::uint32_t model;
};

[[nodiscard]] std::optional<AmdFamilyModel> amd_family_model() {
#if defined(__i386__) || defined(__x86_64__)
    if (x86_vendor() != "AuthenticAMD" || cpuid(0x80000000U, 0U)[0] < 0x80000001U) {
        return std::nullopt;
    }
    const std::uint32_t eax = cpuid(0x80000001U, 0U)[0];
    const std::uint32_t base_family = (eax >> 8U) & 0xfU;
    const std::uint32_t base_model = (eax >> 4U) & 0xfU;
    const std::uint32_t family = base_family == 0xfU ? base_family + ((eax >> 20U) & 0xffU) : base_family;
    const std::uint32_t model = (base_family == 0x6U || base_family == 0xfU)
                                    ? base_model + (((eax >> 16U) & 0xfU) << 4U)
                                    : base_model;
    return AmdFamilyModel{family, model};
#else
    return std::nullopt;
#endif
}

[[nodiscard]] std::string amd_generation() {
    const auto cpu = amd_family_model();
    if (!cpu.has_value()) {
        return "zenx";
    }
    if (cpu->family == 0x1aU) {
        return "zen5";
    }
    if (cpu->family == 0x19U) {
        return cpu->model >= 0x60U ? "zen4" : "zen3";
    }
    if (cpu->family == 0x17U) {
        return cpu->model >= 0x30U ? "zen2" : "zen1";
    }
    return "zenx";
}

[[nodiscard]] std::uint64_t amd_rank(const std::uint64_t l3_bytes, const std::uint64_t max_frequency_khz) {
    constexpr std::uint64_t frequency_bits = 20U;
    return ((l3_bytes / (1024ULL * 1024ULL)) << frequency_bits) |
           ((max_frequency_khz / 1000ULL) & ((1ULL << frequency_bits) - 1ULL));
}

void classify_linux(Snapshot& snapshot) {
    const std::string vendor = x86_vendor();
    if (vendor == "GenuineIntel") {
        bool found_hybrid_type = false;
        for (auto& processor : snapshot.logical_processors) {
            auto type = intel_core_type(processor.logical_processor_id);
            if (!type.has_value()) {
                type = read_integer<std::uint32_t>(cpu_path(processor.logical_processor_id, "/topology/core_type"));
            }
            if (!type.has_value()) {
                continue;
            }
            found_hybrid_type = true;
            processor.performance_rank = *type;
            processor.performance_class_key = "intel_core_type_" + std::to_string(*type);
            const std::string prefix = *type == 0x20U ? "E-core" : (*type == 0x40U ? "P-core" : "core-type-" + std::to_string(*type));
            processor.performance_class_label = prefix + "@cpu" + std::to_string(processor.logical_processor_id);
        }
        if (found_hybrid_type) {
            return;
        }
    }

    const bool is_amd = vendor == "AuthenticAMD";
    const std::string generation = is_amd ? amd_generation() : "zenx";
    for (auto& processor : snapshot.logical_processors) {
        if (is_amd && processor.l3_cache_bytes > 0U) {
            const std::string tier = processor.l3_cache_bytes >= 16ULL * 1024ULL * 1024ULL
                                         ? generation
                                         : generation + "c";
            processor.performance_rank = amd_rank(processor.l3_cache_bytes, processor.max_frequency_khz);
            processor.performance_class_key = "amd_" + tier + "_l3" + std::to_string(processor.l3_cache_bytes) +
                                              "_f" + std::to_string(processor.max_frequency_khz);
            processor.performance_class_label = tier + "@cpu" + std::to_string(processor.logical_processor_id) + " (" +
                                                decimal_label(static_cast<double>(processor.l3_cache_bytes) /
                                                                  (1024.0 * 1024.0),
                                                              1) +
                                                "MiB/" + decimal_label(static_cast<double>(processor.max_frequency_khz) /
                                                                           1'000'000.0,
                                                                       2) +
                                                "GHz)";
        } else if (processor.max_frequency_khz > 0U) {
            processor.performance_rank = processor.max_frequency_khz;
            processor.performance_class_key = "max_freq_" + std::to_string(processor.max_frequency_khz);
            processor.performance_class_label = decimal_label(
                                                    static_cast<double>(processor.max_frequency_khz) / 1'000'000.0, 2) +
                                                "GHz@cpu" + std::to_string(processor.logical_processor_id);
        }
    }
}

[[nodiscard]] Result query_linux() {
    Result result;
    result.snapshot.model_name = cpu_model_linux();
    if (result.snapshot.model_name.empty()) {
        result.snapshot.model_name = "unknown";
    }
    const std::vector<int> cpus = online_cpus_linux();
    bool partial = false;
    for (const int cpu : cpus) {
        LogicalProcessor processor;
        processor.logical_processor_id = cpu;
        processor.core_id = read_integer<int>(cpu_path(cpu, "/topology/core_id")).value_or(kUnknownId);
        processor.package_id = read_integer<int>(cpu_path(cpu, "/topology/physical_package_id")).value_or(kUnknownId);
        processor.numa_node_id = read_numa_node_linux(cpu).value_or(kUnknownId);
        processor.max_frequency_khz =
            read_integer<std::uint64_t>(cpu_path(cpu, "/cpufreq/cpuinfo_max_freq")).value_or(0U);
        processor.l1_data_cache_bytes = read_cache_bytes_linux(cpu, 1, true);
        processor.l2_cache_bytes = read_cache_bytes_linux(cpu, 2, false);
        processor.l3_cache_bytes = read_cache_bytes_linux(cpu, 3, false);
        processor.performance_rank = processor.max_frequency_khz;
        processor.performance_class_key = processor.max_frequency_khz == 0U
                                              ? "default"
                                              : "max_freq_" + std::to_string(processor.max_frequency_khz);
        processor.performance_class_label = "cpu" + std::to_string(cpu);
        partial = partial || processor.core_id < 0 || processor.package_id < 0;
        result.snapshot.logical_processors.push_back(std::move(processor));
    }
    if (result.snapshot.logical_processors.empty()) {
        return unavailable_result("no permitted online logical processors were detected");
    }
    classify_linux(result.snapshot);
    fill_aggregate_fields(result.snapshot);
    partial = partial || !result.snapshot.has_numa_mapping;
    result.status.code = partial ? StatusCode::partial : StatusCode::ok;
    result.status.message = partial ? "CPU topology is partially available from Linux sysfs"
                                    : "CPU topology detected from Linux sysfs and CPUID";
    return result;
}

#endif // __linux__

#if defined(_WIN32)

template <typename Function>
void for_each_set_bit(const std::uint64_t mask, Function&& function) {
    for (std::uint8_t bit = 0U; bit < 64U; ++bit) {
        if ((mask & (std::uint64_t{1} << bit)) != 0U) {
            function(bit);
        }
    }
}

[[nodiscard]] Result query_windows() {
    Result result;
    result.snapshot.model_name = x86_brand();
    if (result.snapshot.model_name.empty()) {
        result.snapshot.model_name = "unknown";
    }
    std::map<std::pair<std::uint16_t, std::uint8_t>, std::size_t> index_by_group_cpu;
    const WORD group_count = GetActiveProcessorGroupCount();
    for (WORD group = 0; group < group_count; ++group) {
        const DWORD cpu_count = GetActiveProcessorCount(group);
        for (DWORD cpu = 0; cpu < cpu_count; ++cpu) {
            LogicalProcessor processor;
            processor.logical_processor_id = static_cast<int>(result.snapshot.logical_processors.size());
            processor.group_id = static_cast<std::uint16_t>(group);
            processor.group_index = static_cast<std::uint8_t>(cpu);
            processor.performance_class_label = "group" + std::to_string(group) + "@cpu" + std::to_string(cpu);
            index_by_group_cpu.emplace(std::make_pair(processor.group_id, processor.group_index),
                                       result.snapshot.logical_processors.size());
            result.snapshot.logical_processors.push_back(std::move(processor));
        }
    }

    DWORD size = 0U;
    if (GetLogicalProcessorInformationEx(RelationAll, nullptr, &size) != 0 || GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        return fallback_result("GetLogicalProcessorInformationEx size query failed");
    }
    std::vector<std::uint8_t> buffer(size);
    auto* information = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
    if (buffer.empty() || GetLogicalProcessorInformationEx(RelationAll, information, &size) == 0) {
        return fallback_result("GetLogicalProcessorInformationEx data query failed");
    }

    int next_core_id = 0;
    int next_package_id = 0;
    for (DWORD offset = 0U; offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) <= size;) {
        auto* entry = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
        if (entry->Size == 0U || offset + entry->Size > size) {
            break;
        }
        if (entry->Relationship == RelationProcessorCore) {
            const auto efficiency = entry->Processor.EfficiencyClass;
            for (WORD index = 0; index < entry->Processor.GroupCount; ++index) {
                const GROUP_AFFINITY& affinity = entry->Processor.GroupMask[index];
                for_each_set_bit(static_cast<std::uint64_t>(affinity.Mask), [&](const std::uint8_t bit) {
                    const auto found = index_by_group_cpu.find({static_cast<std::uint16_t>(affinity.Group), bit});
                    if (found == index_by_group_cpu.end()) {
                        return;
                    }
                    auto& processor = result.snapshot.logical_processors[found->second];
                    processor.core_id = next_core_id;
                    processor.performance_rank = std::numeric_limits<std::uint8_t>::max() - efficiency;
                    processor.performance_class_key = "efficiency_class_" + std::to_string(efficiency);
                    processor.performance_class_label = "eff" + std::to_string(efficiency) + "@group" +
                                                        std::to_string(affinity.Group) + "cpu" + std::to_string(bit);
                });
            }
            ++next_core_id;
        } else if (entry->Relationship == RelationProcessorPackage) {
            for (WORD index = 0; index < entry->Processor.GroupCount; ++index) {
                const GROUP_AFFINITY& affinity = entry->Processor.GroupMask[index];
                for_each_set_bit(static_cast<std::uint64_t>(affinity.Mask), [&](const std::uint8_t bit) {
                    const auto found = index_by_group_cpu.find({static_cast<std::uint16_t>(affinity.Group), bit});
                    if (found != index_by_group_cpu.end()) {
                        result.snapshot.logical_processors[found->second].package_id = next_package_id;
                    }
                });
            }
            ++next_package_id;
        } else if (entry->Relationship == RelationNumaNode) {
            const auto& affinity = entry->NumaNode.GroupMask;
            for_each_set_bit(static_cast<std::uint64_t>(affinity.Mask), [&](const std::uint8_t bit) {
                const auto found = index_by_group_cpu.find({static_cast<std::uint16_t>(affinity.Group), bit});
                if (found != index_by_group_cpu.end()) {
                    result.snapshot.logical_processors[found->second].numa_node_id =
                        static_cast<int>(entry->NumaNode.NodeNumber);
                }
            });
        } else if (entry->Relationship == RelationCache &&
                   (entry->Cache.Type == CacheData || entry->Cache.Type == CacheUnified)) {
            const auto& affinity = entry->Cache.GroupMask;
            for_each_set_bit(static_cast<std::uint64_t>(affinity.Mask), [&](const std::uint8_t bit) {
                const auto found = index_by_group_cpu.find({static_cast<std::uint16_t>(affinity.Group), bit});
                if (found == index_by_group_cpu.end()) {
                    return;
                }
                auto& processor = result.snapshot.logical_processors[found->second];
                if (entry->Cache.Level == 1U) {
                    processor.l1_data_cache_bytes = std::max(processor.l1_data_cache_bytes,
                                                             static_cast<std::uint64_t>(entry->Cache.CacheSize));
                } else if (entry->Cache.Level == 2U) {
                    processor.l2_cache_bytes = std::max(processor.l2_cache_bytes,
                                                        static_cast<std::uint64_t>(entry->Cache.CacheSize));
                } else if (entry->Cache.Level == 3U) {
                    processor.l3_cache_bytes = std::max(processor.l3_cache_bytes,
                                                        static_cast<std::uint64_t>(entry->Cache.CacheSize));
                }
            });
        }
        offset += entry->Size;
    }
    for (auto& processor : result.snapshot.logical_processors) {
        if (processor.core_id < 0) {
            processor.core_id = processor.logical_processor_id;
        }
        if (processor.package_id < 0) {
            processor.package_id = 0;
        }
    }
    fill_aggregate_fields(result.snapshot);
    result.status.code = result.snapshot.has_numa_mapping ? StatusCode::ok : StatusCode::partial;
    result.status.message = "CPU topology detected with GetLogicalProcessorInformationEx";
    return result;
}

#endif // _WIN32

} // namespace

Result query_snapshot() noexcept {
    try {
#if defined(__linux__)
        return query_linux();
#elif defined(_WIN32)
        return query_windows();
#else
        return fallback_result("CPU topology affinity is unsupported on this platform");
#endif
    } catch (...) {
        return fallback_result("CPU topology query failed; fallback data is used");
    }
}

std::vector<CoreClass> build_core_classes(const Snapshot& snapshot) {
    std::map<std::string, CoreClass> grouped;
    for (const auto& processor : snapshot.logical_processors) {
        const std::string key = processor.performance_class_key.empty() ? "default" : processor.performance_class_key;
        const std::string label = processor.performance_class_label.empty()
                                      ? "cpu" + std::to_string(processor.logical_processor_id)
                                      : processor.performance_class_label;
        auto [iterator, inserted] = grouped.try_emplace(key, CoreClass{
                                                                .key = key,
                                                                .label = label,
                                                                .logical_processor = processor,
                                                                .performance_rank = processor.performance_rank,
                                                            });
        if (!inserted && (processor.performance_rank > iterator->second.performance_rank ||
                          (processor.performance_rank == iterator->second.performance_rank &&
                           processor.logical_processor_id < iterator->second.logical_processor.logical_processor_id))) {
            iterator->second.label = label;
            iterator->second.logical_processor = processor;
            iterator->second.performance_rank = processor.performance_rank;
        }
    }

    std::vector<CoreClass> classes;
    classes.reserve(grouped.size());
    for (auto& [key, core_class] : grouped) {
        (void)key;
        classes.push_back(std::move(core_class));
    }
    std::sort(classes.begin(), classes.end(), [](const CoreClass& left, const CoreClass& right) {
        if (left.performance_rank != right.performance_rank) {
            return left.performance_rank > right.performance_rank;
        }
        return left.logical_processor.logical_processor_id < right.logical_processor.logical_processor_id;
    });
    return classes;
}

ScopedThreadAffinity::ScopedThreadAffinity(const LogicalProcessor& processor) noexcept {
#if defined(__linux__)
    if (processor.logical_processor_id < 0 || processor.logical_processor_id >= CPU_SETSIZE) {
        return;
    }
    cpu_set_t previous;
    CPU_ZERO(&previous);
    if (pthread_getaffinity_np(pthread_self(), sizeof(previous), &previous) != 0) {
        return;
    }
    restore_previous_ = true;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(static_cast<unsigned>(cpu), &previous)) {
            previous_cpu_ids_.push_back(cpu);
        }
    }
    cpu_set_t target;
    CPU_ZERO(&target);
    CPU_SET(static_cast<unsigned>(processor.logical_processor_id), &target);
    pinned_ = pthread_setaffinity_np(pthread_self(), sizeof(target), &target) == 0;
#elif defined(_WIN32)
    if (processor.group_index >= 64U) {
        return;
    }
    GROUP_AFFINITY target{};
    target.Group = static_cast<WORD>(processor.group_id);
    target.Mask = static_cast<KAFFINITY>(std::uint64_t{1} << processor.group_index);
    GROUP_AFFINITY previous{};
    if (SetThreadGroupAffinity(GetCurrentThread(), &target, &previous) != 0) {
        previous_group_id_ = static_cast<std::uint16_t>(previous.Group);
        previous_group_mask_ = static_cast<std::uint64_t>(previous.Mask);
        restore_previous_ = true;
        pinned_ = true;
    }
#else
    (void)processor;
#endif
}

ScopedThreadAffinity::~ScopedThreadAffinity() {
#if defined(__linux__)
    if (!restore_previous_ || previous_cpu_ids_.empty()) {
        return;
    }
    cpu_set_t previous;
    CPU_ZERO(&previous);
    for (const int cpu : previous_cpu_ids_) {
        if (cpu >= 0 && cpu < CPU_SETSIZE) {
            CPU_SET(static_cast<unsigned>(cpu), &previous);
        }
    }
    (void)pthread_setaffinity_np(pthread_self(), sizeof(previous), &previous);
#elif defined(_WIN32)
    if (!restore_previous_) {
        return;
    }
    GROUP_AFFINITY previous{};
    previous.Group = static_cast<WORD>(previous_group_id_);
    previous.Mask = static_cast<KAFFINITY>(previous_group_mask_);
    (void)SetThreadGroupAffinity(GetCurrentThread(), &previous, nullptr);
#endif
}

} // namespace uni::simd::benchmark::cpu_topology
