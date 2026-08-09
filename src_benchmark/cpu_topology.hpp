#pragma once

#include <cstdint>
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
    int logical_processor_id = -1;
    int core_id = -1;
    int package_id = -1;
    int numa_node_id = -1;
    std::uint64_t max_frequency_khz = 0U;
    std::uint64_t l1_data_cache_bytes = 0U;
    std::uint64_t l2_cache_bytes = 0U;
    std::uint64_t l3_cache_bytes = 0U;
    std::uint16_t group_id = 0U;
    std::uint8_t group_index = 0U;
    std::uint64_t performance_rank = 0U;
    std::string performance_class_key = "default";
    std::string performance_class_label = "unbound";
};

struct Snapshot {
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
    explicit ScopedThreadAffinity(const LogicalProcessor& logical_processor) noexcept;
    ~ScopedThreadAffinity();

    ScopedThreadAffinity(const ScopedThreadAffinity&) = delete;
    ScopedThreadAffinity& operator=(const ScopedThreadAffinity&) = delete;

    [[nodiscard]] ThreadAffinityStatus status() const noexcept { return status_; }
    [[nodiscard]] bool can_run() const noexcept { return status_ != ThreadAffinityStatus::failed; }

private:
    ThreadAffinityStatus status_ = ThreadAffinityStatus::failed;
    bool restore_previous_ = false;
    std::uint16_t previous_group_id_ = 0U;
    std::uint64_t previous_group_mask_ = 0U;
    int previous_affinity_tag_ = 0;
    std::vector<int> previous_cpu_ids_;
};

} // namespace uni::simd::benchmark::cpu_topology
