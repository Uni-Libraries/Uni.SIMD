#pragma once

#include <uni_sysinfo_cpu.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace uni::simd::benchmark::cpu_topology {

enum class StatusCode : std::uint8_t {
    ok,
    partial,
    unavailable,
};

struct Status {
    StatusCode code = StatusCode::unavailable;
    std::string message;

    [[nodiscard]] bool has_data() const noexcept {
        return code == StatusCode::ok || code == StatusCode::partial;
    }
};

struct LogicalProcessor {
    std::size_t snapshot_index = 0U;
    std::size_t os_cpu_id = 0U;
    std::uint16_t os_group = 0U;
    int logical_processor_id = -1;
    int core_id = -1;
    int package_id = -1;
    int numa_node_id = -1;
    std::uint64_t max_frequency_khz = 0U;
    std::uint64_t l1_data_cache_bytes = 0U;
    std::uint64_t l2_cache_bytes = 0U;
    std::uint64_t l3_cache_bytes = 0U;
    std::uint64_t performance_rank = 0U;
    std::string performance_class_key = "default";
    std::string performance_class_label = "unbound";
};

struct Snapshot {
    std::shared_ptr<uni_sysinfo_cpu_snapshot_t> native;
    std::string model_name = "unknown";
    std::vector<LogicalProcessor> logical_processors;
    std::uint32_t logical_processor_count = 0U;
    std::uint32_t physical_core_count = 0U;
    std::uint32_t package_count = 0U;
    std::uint32_t numa_node_count = 0U;
    bool has_core_mapping = false;
    bool has_package_mapping = false;
    bool has_numa_mapping = false;
};

struct Result {
    Status status;
    Snapshot snapshot;
};

struct CoreClass {
    std::string key = "default";
    std::string label = "unbound";
    LogicalProcessor logical_processor;
    std::uint64_t performance_rank = 0U;
};

enum class ThreadAffinityStatus : std::uint8_t {
    failed,
    applied,
    unsupported,
};

[[nodiscard]] Result query_snapshot() noexcept;
[[nodiscard]] std::vector<CoreClass> build_core_classes(const Snapshot& snapshot);

class ScopedThreadAffinity final {
public:
    ScopedThreadAffinity(const Snapshot& snapshot, const LogicalProcessor& logical_processor) noexcept;
    ~ScopedThreadAffinity();

    ScopedThreadAffinity(const ScopedThreadAffinity&) = delete;
    ScopedThreadAffinity& operator=(const ScopedThreadAffinity&) = delete;

    [[nodiscard]] ThreadAffinityStatus status() const noexcept { return status_; }
    [[nodiscard]] bool can_run() const noexcept { return status_ != ThreadAffinityStatus::failed; }

private:
    ThreadAffinityStatus status_ = ThreadAffinityStatus::failed;
    uni_sysinfo_cpu_affinity_t* previous_affinity_ = nullptr;
    bool restore_platform_affinity_ = false;
    std::uint16_t previous_group_id_ = 0U;
    std::uint64_t previous_group_mask_ = 0U;
};

} // namespace uni::simd::benchmark::cpu_topology
