#include "cpu_topology.hpp"

#include <uni_sysinfo_typedefs.h>

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

namespace uni::simd::benchmark::cpu_topology {
namespace {

constexpr int kUnknownId = -1;

[[nodiscard]] std::string_view result_name(const uni_sysinfo_result_e result) noexcept {
    switch (result) {
    case UNI_SYSINFO_RESULT_OK:
        return "ok";
    case UNI_SYSINFO_RESULT_INVALID_ARGUMENT:
        return "invalid argument";
    case UNI_SYSINFO_RESULT_OUT_OF_RANGE:
        return "out of range";
    case UNI_SYSINFO_RESULT_BUFFER_TOO_SMALL:
        return "buffer too small";
    case UNI_SYSINFO_RESULT_NOT_SUPPORTED:
        return "not supported";
    case UNI_SYSINFO_RESULT_SYSTEM_ERROR:
        return "system error";
    case UNI_SYSINFO_RESULT_UNKNOWN:
    default:
        return "unknown error";
    }
}

[[nodiscard]] bool has_field(const uni_sysinfo_cpu_logical_processor_t& processor,
                             const uni_sysinfo_cpu_valid_fields_t field) noexcept {
    return (processor.valid_fields & field) != 0U;
}

[[nodiscard]] int to_id(const std::size_t value) noexcept {
    return value <= static_cast<std::size_t>(std::numeric_limits<int>::max()) ? static_cast<int>(value) : kUnknownId;
}

[[nodiscard]] std::string core_type_name(const uni_sysinfo_cpu_core_type_e type) {
    switch (type) {
    case UNI_SYSINFO_CPU_CORE_TYPE_PERFORMANCE:
        return "performance";
    case UNI_SYSINFO_CPU_CORE_TYPE_EFFICIENCY:
        return "efficiency";
    case UNI_SYSINFO_CPU_CORE_TYPE_LOW_POWER:
        return "low-power";
    case UNI_SYSINFO_CPU_CORE_TYPE_UNKNOWN:
    default:
        return "unknown";
    }
}

void set_performance_class(LogicalProcessor& output, const uni_sysinfo_cpu_logical_processor_t& processor) {
    const std::string cpu_suffix = "@cpu" + std::to_string(output.logical_processor_id);
    if (has_field(processor, UNI_SYSINFO_CPU_VALID_PERFORMANCE_CLASS)) {
        output.performance_class_key = "performance_" + std::to_string(processor.performance_rank) + "_raw_" +
                                       std::to_string(processor.raw_efficiency_class);
        if (has_field(processor, UNI_SYSINFO_CPU_VALID_CORE_TYPE)) {
            output.performance_class_key += "_type_" + std::to_string(processor.topology.core_type);
            output.performance_class_label = core_type_name(processor.topology.core_type) + cpu_suffix;
        } else {
            output.performance_class_key += "_l2_" + std::to_string(processor.cache.l2_size) + "_l3_" +
                                            std::to_string(processor.cache.l3_size);
            output.performance_class_label = "class-" + std::to_string(processor.raw_efficiency_class) + cpu_suffix;
        }
        return;
    }
    if (has_field(processor, UNI_SYSINFO_CPU_VALID_CORE_TYPE)) {
        const std::string type = core_type_name(processor.topology.core_type);
        output.performance_class_key = "core_type_" + std::to_string(processor.topology.core_type);
        output.performance_class_label = type + cpu_suffix;
        return;
    }
    output.performance_class_label = "default" + cpu_suffix;
}

void fill_aggregate_fields(Snapshot& snapshot) {
    std::set<std::pair<int, int>> physical_cores;
    std::set<int> packages;
    std::set<int> numa_nodes;
    bool all_have_core = !snapshot.logical_processors.empty();
    bool all_have_package = !snapshot.logical_processors.empty();

    snapshot.logical_processor_count = static_cast<std::uint32_t>(std::min<std::size_t>(
        snapshot.logical_processors.size(), std::numeric_limits<std::uint32_t>::max()));
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
        std::min<std::size_t>(physical_cores.empty() ? snapshot.logical_processors.size() : physical_cores.size(),
                              std::numeric_limits<std::uint32_t>::max()));
    snapshot.package_count = static_cast<std::uint32_t>(std::min<std::size_t>(
        packages.empty() ? (snapshot.logical_processors.empty() ? 0U : 1U) : packages.size(),
        std::numeric_limits<std::uint32_t>::max()));
    snapshot.numa_node_count = static_cast<std::uint32_t>(
        std::min<std::size_t>(numa_nodes.size(), std::numeric_limits<std::uint32_t>::max()));
}

} // namespace

Result query_snapshot() noexcept {
    Result result;
    try {
        uni_sysinfo_cpu_snapshot_t* raw_snapshot = nullptr;
        const auto create_status = uni_sysinfo_cpu_snapshot_create(&raw_snapshot);
        if (create_status != UNI_SYSINFO_RESULT_OK) {
            result.status = {
                .code = StatusCode::unavailable,
                .message = "Uni.SysInfo snapshot creation failed: " + std::string(result_name(create_status)),
            };
            return result;
        }
        result.snapshot.native = {raw_snapshot, &uni_sysinfo_cpu_snapshot_destroy};

        std::size_t logical_count = 0U;
        const auto count_status = uni_sysinfo_cpu_snapshot_logical_count(raw_snapshot, &logical_count);
        if (count_status != UNI_SYSINFO_RESULT_OK) {
            result.status = {
                .code = StatusCode::unavailable,
                .message = "Uni.SysInfo logical CPU query failed: " + std::string(result_name(count_status)),
            };
            return result;
        }

        bool partial = false;
        for (std::size_t index = 0U; index < logical_count; ++index) {
            uni_sysinfo_cpu_logical_processor_t processor{};
            const auto processor_status = uni_sysinfo_cpu_snapshot_logical_processor(raw_snapshot, index, &processor);
            if (processor_status != UNI_SYSINFO_RESULT_OK) {
                partial = true;
                continue;
            }
            const bool allowed_set_valid = has_field(processor, UNI_SYSINFO_CPU_VALID_ALLOWED_SET);
            if (!processor.online || (allowed_set_valid && !processor.allowed)) {
                continue;
            }

            LogicalProcessor output;
            output.snapshot_index = index;
#if defined(_WIN32)
            output.logical_processor_id = to_id(index);
#else
            output.logical_processor_id = has_field(processor, UNI_SYSINFO_CPU_VALID_NATIVE_ID)
                                              ? to_id(processor.os_cpu_id)
                                              : to_id(index);
#endif
            output.os_cpu_id = processor.os_cpu_id;
            output.os_group = processor.os_group;
            output.package_id = has_field(processor, UNI_SYSINFO_CPU_VALID_TOPOLOGY)
                                    ? to_id(processor.package_idx)
                                    : kUnknownId;
            output.core_id = has_field(processor, UNI_SYSINFO_CPU_VALID_TOPOLOGY)
                                 ? to_id(processor.topology.hw_core_idx)
                                 : kUnknownId;
            output.numa_node_id = has_field(processor, UNI_SYSINFO_CPU_VALID_NUMA_NODE)
                                      ? to_id(processor.numa_node_idx)
                                      : kUnknownId;
            output.max_frequency_khz = has_field(processor, UNI_SYSINFO_CPU_VALID_MAX_FREQUENCY)
                                           ? processor.max_frequency_hz / 1000U
                                           : 0U;
            if (has_field(processor, UNI_SYSINFO_CPU_VALID_CACHE)) {
                output.l1_data_cache_bytes = processor.cache.l1d_size;
                output.l2_cache_bytes = processor.cache.l2_size;
                output.l3_cache_bytes = processor.cache.l3_size;
            }
            output.performance_rank = processor.performance_rank;
            set_performance_class(output, processor);

            if (result.snapshot.model_name == "unknown" && has_field(processor, UNI_SYSINFO_CPU_VALID_NAME) &&
                processor.name[0] != '\0') {
                result.snapshot.model_name = processor.name;
            }
            partial = partial || !allowed_set_valid || !has_field(processor, UNI_SYSINFO_CPU_VALID_TOPOLOGY) ||
                      !has_field(processor, UNI_SYSINFO_CPU_VALID_NATIVE_ID) ||
                      !has_field(processor, UNI_SYSINFO_CPU_VALID_CACHE) ||
                      !has_field(processor, UNI_SYSINFO_CPU_VALID_MAX_FREQUENCY) ||
                      !has_field(processor, UNI_SYSINFO_CPU_VALID_NUMA_NODE);
            result.snapshot.logical_processors.push_back(std::move(output));
        }

        if (result.snapshot.logical_processors.empty()) {
            result.status = {
                .code = StatusCode::unavailable,
                .message = "Uni.SysInfo did not report an online logical processor available to this process",
            };
            return result;
        }
        fill_aggregate_fields(result.snapshot);
        result.status = {
            .code = partial ? StatusCode::partial : StatusCode::ok,
            .message = partial ? "CPU topology is partially available from Uni.SysInfo"
                               : "CPU topology detected by Uni.SysInfo",
        };
        return result;
    } catch (...) {
        result.status = {
            .code = StatusCode::unavailable,
            .message = "Uni.SysInfo CPU topology adaptation failed",
        };
        return result;
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

ScopedThreadAffinity::ScopedThreadAffinity(const Snapshot& snapshot, const LogicalProcessor& processor) noexcept {
    if (!snapshot.native) {
        return;
    }
    const auto result = uni_sysinfo_cpu_affinity_apply(snapshot.native.get(), processor.snapshot_index,
                                                        &previous_affinity_);
    if (result == UNI_SYSINFO_RESULT_OK) {
        status_ = ThreadAffinityStatus::applied;
#if defined(__APPLE__)
    } else if (result == UNI_SYSINFO_RESULT_NOT_SUPPORTED) {
        status_ = ThreadAffinityStatus::unsupported;
#elif defined(_WIN32)
    } else if (result == UNI_SYSINFO_RESULT_NOT_SUPPORTED && processor.os_cpu_id < 64U) {
        GROUP_AFFINITY target{};
        target.Group = static_cast<WORD>(processor.os_group);
        target.Mask = static_cast<KAFFINITY>(std::uint64_t{1} << processor.os_cpu_id);
        GROUP_AFFINITY previous{};
        if (SetThreadGroupAffinity(GetCurrentThread(), &target, &previous) != 0) {
            previous_group_id_ = static_cast<std::uint16_t>(previous.Group);
            previous_group_mask_ = static_cast<std::uint64_t>(previous.Mask);
            restore_platform_affinity_ = true;
            status_ = ThreadAffinityStatus::applied;
        }
#endif
    }
}

ScopedThreadAffinity::~ScopedThreadAffinity() {
    if (previous_affinity_ != nullptr) {
        (void)uni_sysinfo_cpu_affinity_restore(previous_affinity_);
        uni_sysinfo_cpu_affinity_destroy(previous_affinity_);
    }
#if defined(_WIN32)
    if (restore_platform_affinity_) {
        GROUP_AFFINITY previous{};
        previous.Group = static_cast<WORD>(previous_group_id_);
        previous.Mask = static_cast<KAFFINITY>(previous_group_mask_);
        (void)SetThreadGroupAffinity(GetCurrentThread(), &previous, nullptr);
    }
#endif
}

} // namespace uni::simd::benchmark::cpu_topology
