#include "qpsk_carrier_analyzer_internal.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace uni::simd::kernels {

[[nodiscard]] constexpr bool ValidBackend(const uni_simd_backend_e backend) noexcept { return backend <= UNI_SIMD_BACKEND_AARCH64_NEON; }

[[nodiscard]] constexpr bool NaturallyAligned(const void* const pointer) noexcept { return reinterpret_cast<std::uintptr_t>(pointer) % alignof(float) == 0U; }

[[nodiscard]] static bool Overlaps(const void* const left, const std::size_t left_bytes,
                                   const void* const right, const std::size_t right_bytes) noexcept {
    if (left_bytes == 0U || right_bytes == 0U) {
        return false;
    }
    const auto left_begin = reinterpret_cast<std::uintptr_t>(left);
    const auto right_begin = reinterpret_cast<std::uintptr_t>(right);
    return left_begin <= right_begin ? right_begin - left_begin < left_bytes
                                     : left_begin - right_begin < right_bytes;
}

uni_simd_result_e QpskCarrierAnalyzerInitialize(
    uni_simd_qpsk_carrier_analyzer_t& analyzer,
    const uni_simd_qpsk_carrier_analyzer_config_t& config,
    const uni_simd_backend_e requested_backend,
    const uni_simd_math_mode_e math_mode,
    const bool prefer_energy_efficiency) noexcept {
    if (config.descriptor_size != UNI_SIMD_QPSK_CARRIER_ANALYZER_CONFIG_DESCRIPTOR_SIZE ||
        !std::isfinite(config.magnitude_epsilon) || config.magnitude_epsilon <= 0.0f || !ValidBackend(requested_backend) ||
        math_mode > UNI_SIMD_MATH_DETERMINISTIC) {
        return UNI_SIMD_RESULT_INVALID_ARGUMENT;
    }
    analyzer = {};
    analyzer.magnitude_epsilon = config.magnitude_epsilon;
    analyzer.analyze = &QpskCarrierAnalyzer_generic;

    if (math_mode != UNI_SIMD_MATH_DETERMINISTIC) {
        const auto& capabilities = uni::simd::capabilities();
        bool selected = requested_backend == UNI_SIMD_BACKEND_GENERIC;

#if UNI_SIMD_HAVE_AVX2_FMA
        const bool avx2_available = capabilities.avx2 && capabilities.fma;
        if (avx2_available && (requested_backend == UNI_SIMD_BACKEND_AUTOMATIC || requested_backend == UNI_SIMD_BACKEND_X86_AVX2_FMA ||
                                requested_backend == UNI_SIMD_BACKEND_X86_AVX512)) {
            analyzer.analyze = &QpskCarrierAnalyzer_avx2_fma;
            analyzer.backend = UNI_SIMD_BACKEND_X86_AVX2_FMA;
            selected = true;
        }
#endif
#if UNI_SIMD_HAVE_AVX512F
        if (capabilities.avx512f && requested_backend == UNI_SIMD_BACKEND_X86_AVX512) {
            analyzer.analyze = &QpskCarrierAnalyzer_avx512f;
            analyzer.backend = UNI_SIMD_BACKEND_X86_AVX512;
            selected = true;
        }
        if (capabilities.avx512f && requested_backend == UNI_SIMD_BACKEND_AUTOMATIC &&
            !prefer_energy_efficiency) {
            analyzer.analyze = &QpskCarrierAnalyzer_avx512f;
            analyzer.backend = UNI_SIMD_BACKEND_X86_AVX512;
            selected = true;
        }
#endif
#if UNI_SIMD_HAVE_NEON
        if (capabilities.neon && (requested_backend == UNI_SIMD_BACKEND_AUTOMATIC || requested_backend == UNI_SIMD_BACKEND_AARCH64_NEON)) {
            analyzer.analyze = &QpskCarrierAnalyzer_neon;
            analyzer.backend = UNI_SIMD_BACKEND_AARCH64_NEON;
            selected = true;
        }
#endif
        if (!selected && requested_backend != UNI_SIMD_BACKEND_AUTOMATIC) {
            return UNI_SIMD_RESULT_UNSUPPORTED_BACKEND;
        }
    }

    return UNI_SIMD_RESULT_SUCCESS;
}

uni_simd_result_e QpskCarrierAnalyzerReset(uni_simd_qpsk_carrier_analyzer_t& analyzer) noexcept {
    analyzer.previous_fourth_real = 0.0f;
    analyzer.previous_fourth_imag = 0.0f;
    analyzer.previous_fourth_valid = false;
    return UNI_SIMD_RESULT_SUCCESS;
}

uni_simd_result_e QpskCarrierAnalyzerExecute(
    uni_simd_qpsk_carrier_analyzer_t& analyzer,
    const uni_simd_qpsk_carrier_analyzer_block_t& block,
    uni_simd_qpsk_carrier_analyzer_result_t& result) noexcept {
    if (block.descriptor_size != UNI_SIMD_QPSK_CARRIER_ANALYZER_BLOCK_DESCRIPTOR_SIZE ||
        (block.sample_count != 0U && block.samples == nullptr) ||
        block.sample_count > std::numeric_limits<std::size_t>::max() / (UNI_SIMD_CF32_COMPONENT_COUNT * sizeof(float)) ||
        (block.sample_count != 0U && !NaturallyAligned(block.samples)) ||
        Overlaps(block.samples, block.sample_count * UNI_SIMD_CF32_COMPONENT_COUNT * sizeof(float),
                 &result, sizeof(result))) {
        return UNI_SIMD_RESULT_INVALID_ARGUMENT;
    }

    result = {};
    result.descriptor_size = UNI_SIMD_QPSK_CARRIER_ANALYZER_RESULT_DESCRIPTOR_SIZE;
    if (block.sample_count != 0U) {
        analyzer.analyze(analyzer, block.samples, block.sample_count, result);
    }
    return UNI_SIMD_RESULT_SUCCESS;
}

} // namespace uni::simd::kernels
