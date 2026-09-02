#include "common/api_internal.hpp"

#include <uni_sysinfo_cpu_aarch64.h>
#include <uni_sysinfo_cpu_x86.h>

namespace uni::simd {
namespace {

[[nodiscard]] Capabilities detect_capabilities() noexcept {
    Capabilities result{};
    uni_sysinfo_cpu_snapshot_t* snapshot = nullptr;
    if (uni_sysinfo_cpu_snapshot_create(&snapshot) != UNI_SYSINFO_RESULT_OK) {
        return result;
    }

    uni_sysinfo_cpu_x86_features_t x86_features{};
    if (uni_sysinfo_cpu_x86_features_common(snapshot, &x86_features) == UNI_SYSINFO_RESULT_OK) {
        result.sse2 = (x86_features & UNI_SYSINFO_CPU_X86_FEATURE_SSE2) != 0U;
        result.avx2 = (x86_features & UNI_SYSINFO_CPU_X86_FEATURE_AVX2) != 0U;
        result.fma = (x86_features & UNI_SYSINFO_CPU_X86_FEATURE_FMA) != 0U;
        result.avx512f = (x86_features & UNI_SYSINFO_CPU_X86_FEATURE_AVX512F) != 0U;
        result.avx512bw = (x86_features & UNI_SYSINFO_CPU_X86_FEATURE_AVX512BW) != 0U;
    }

    uni_sysinfo_cpu_aarch64_features_t aarch64_features{};
    if (uni_sysinfo_cpu_aarch64_features_common(snapshot, &aarch64_features) == UNI_SYSINFO_RESULT_OK) {
        result.neon = (aarch64_features & UNI_SYSINFO_CPU_AARCH64_FEATURE_NEON) != 0U;
    }
    uni_sysinfo_cpu_snapshot_destroy(snapshot);
    return result;
}

} // namespace

const Capabilities& capabilities() noexcept {
    static const Capabilities detected = detect_capabilities();
    return detected;
}

} // namespace uni::simd
