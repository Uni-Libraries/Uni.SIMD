#include "viterbi_internal.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>

namespace uni::simd::kernels {

namespace {

[[nodiscard]] bool ValidPolynomial(const std::uint32_t polynomial) noexcept { return polynomial != 0U && polynomial <= 0x7FU; }

/** Parity of the code bit a polynomial produces when `register_bits` (newest input in bit 6) are in the encoder. */
[[nodiscard]] std::uint16_t ExpectedBit(const std::uint32_t polynomial, const std::uint32_t register_bits) noexcept {
    return (std::popcount(polynomial & register_bits) & 1) != 0 ? 255U : 0U;
}

} // namespace

void ViterbiRenormalize(ViterbiState& state) noexcept {
    const std::uint16_t minimum = *std::min_element(std::begin(state.metrics), std::end(state.metrics));
    for (auto& metric : state.metrics) {
        metric = static_cast<std::uint16_t>(metric - minimum);
    }
    state.steps_since_renormalize = 0U;
}

uni_simd_result_e ViterbiK7R2Initialize(uni_simd_viterbi_k7r2_t& kernel, const uni_simd_viterbi_k7r2_config_t& config,
                                        const uni_simd_backend_e requested_backend) noexcept {
    if (config.descriptor_size != UNI_SIMD_VITERBI_K7R2_CONFIG_DESCRIPTOR_SIZE || !ValidPolynomial(config.polynomial_a) ||
        !ValidPolynomial(config.polynomial_b) || config.start_penalty > 1024U || requested_backend > UNI_SIMD_BACKEND_AARCH64_NEON) {
        return UNI_SIMD_RESULT_INVALID_ARGUMENT;
    }
    kernel = {};
    kernel.config = config;
    auto& initial = kernel.initial_state;
    for (std::size_t state = 0U; state < kViterbiStates; ++state) {
        initial.metrics[state] = config.known_start_state != 0U && state == 0U ? 0U : static_cast<std::uint16_t>(config.start_penalty);
    }
    // Butterfly i joins old states i and i + 32 into new states 2i and 2i + 1. Taking old state
    // i (oldest bit 0) with a new 0 bit puts the register i << 1 into the encoder: with the
    // newest bit in bit 0 of a state, the encoder register holds the state bits reversed.
    for (std::size_t butterfly = 0U; butterfly < kViterbiButterflies; ++butterfly) {
        std::uint32_t register_bits = 0U;
        const std::uint32_t state_bits = static_cast<std::uint32_t>(butterfly) << 1U; // new bit 0 in bit 0
        for (std::uint32_t bit = 0U; bit < 7U; ++bit) {
            register_bits |= ((state_bits >> bit) & 1U) << (6U - bit);
        }
        initial.expected_a[butterfly] = ExpectedBit(config.polynomial_a, register_bits);
        initial.expected_b[butterfly] = ExpectedBit(config.polynomial_b, register_bits);
    }
    kernel.state = initial;

    kernel.run = &ViterbiK7R2_generic;
    kernel.backend = UNI_SIMD_BACKEND_GENERIC;
#if UNI_SIMD_HAVE_AVX2
    const bool avx2_requested = requested_backend == UNI_SIMD_BACKEND_AUTOMATIC || requested_backend == UNI_SIMD_BACKEND_X86_AVX2 ||
                                requested_backend == UNI_SIMD_BACKEND_X86_AVX2_FMA || requested_backend == UNI_SIMD_BACKEND_X86_AVX512;
    if (uni::simd::capabilities().avx2 && avx2_requested) {
        kernel.run = &ViterbiK7R2_avx2;
        kernel.backend = UNI_SIMD_BACKEND_X86_AVX2;
    }
#endif
    if (requested_backend != UNI_SIMD_BACKEND_AUTOMATIC && requested_backend != UNI_SIMD_BACKEND_GENERIC && kernel.backend == UNI_SIMD_BACKEND_GENERIC) {
        return UNI_SIMD_RESULT_UNSUPPORTED_BACKEND;
    }
    return UNI_SIMD_RESULT_SUCCESS;
}

uni_simd_result_e ViterbiK7R2Reset(uni_simd_viterbi_k7r2_t& kernel) noexcept {
    kernel.state = kernel.initial_state;
    return UNI_SIMD_RESULT_SUCCESS;
}

uni_simd_result_e ViterbiK7R2Execute(uni_simd_viterbi_k7r2_t& kernel, const uni_simd_viterbi_k7r2_block_t& block,
                                     uni_simd_viterbi_k7r2_result_t& result) noexcept {
    if (block.descriptor_size != UNI_SIMD_VITERBI_K7R2_BLOCK_DESCRIPTOR_SIZE || result.descriptor_size != UNI_SIMD_VITERBI_K7R2_RESULT_DESCRIPTOR_SIZE ||
        (block.steps != 0U && (block.soft == nullptr || block.decisions == nullptr)) ||
        reinterpret_cast<std::uintptr_t>(block.decisions) % alignof(std::uint64_t) != 0U) {
        return UNI_SIMD_RESULT_INVALID_ARGUMENT;
    }
    if (block.steps > std::numeric_limits<std::size_t>::max() / 2U) {
        return UNI_SIMD_RESULT_INVALID_SIZE;
    }
    if (block.steps != 0U) {
        kernel.run(kernel.state, block.soft, block.steps, block.decisions);
    }
    const auto* const metrics = kernel.state.metrics;
    result.best_state = static_cast<std::uint32_t>(std::min_element(metrics, metrics + kViterbiStates) - metrics);
    return UNI_SIMD_RESULT_SUCCESS;
}

} // namespace uni::simd::kernels
