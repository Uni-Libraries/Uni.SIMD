#pragma once

#include "common/api_internal.hpp"
#include "uni_simd_typedefs.h"

#include <cstddef>
#include <cstdint>

struct uni_simd_viterbi_k7r2_t;

namespace uni::simd::kernels {

inline constexpr std::size_t kViterbiStates = 64U;
inline constexpr std::size_t kViterbiButterflies = kViterbiStates / 2U;

/**
 * Steps between renormalisations of the path metrics. A metric grows by at most 63 per step,
 * so 16-bit metrics keep well clear of overflow in between.
 */
inline constexpr std::size_t kViterbiRenormalizePeriod = 128U;

/** Path metrics and the expected code bits of every butterfly. */
struct ViterbiState final {
    alignas(64) std::uint16_t metrics[kViterbiStates];
    /** Expected bits (0 / 255) of butterfly i for polynomial a and b, i.e. of old state i with a new 0 bit. */
    alignas(64) std::uint16_t expected_a[kViterbiButterflies];
    alignas(64) std::uint16_t expected_b[kViterbiButterflies];
    std::size_t steps_since_renormalize{};
};

using ViterbiRun = void (*)(ViterbiState&, const std::uint8_t* soft, std::size_t steps, std::uint64_t* decisions) noexcept;

void ViterbiK7R2_generic(ViterbiState& state, const std::uint8_t* soft, std::size_t steps, std::uint64_t* decisions) noexcept;
#if UNI_SIMD_HAVE_AVX2
void ViterbiK7R2_avx2(ViterbiState& state, const std::uint8_t* soft, std::size_t steps, std::uint64_t* decisions) noexcept;
#endif

/** Subtract the smallest metric from all of them. */
void ViterbiRenormalize(ViterbiState& state) noexcept;

[[nodiscard]] uni_simd_result_e ViterbiK7R2Initialize(uni_simd_viterbi_k7r2_t& kernel, const uni_simd_viterbi_k7r2_config_t& config,
                                                      uni_simd_backend_e requested_backend) noexcept;
[[nodiscard]] uni_simd_result_e ViterbiK7R2Reset(uni_simd_viterbi_k7r2_t& kernel) noexcept;
[[nodiscard]] uni_simd_result_e ViterbiK7R2Execute(uni_simd_viterbi_k7r2_t& kernel, const uni_simd_viterbi_k7r2_block_t& block,
                                                   uni_simd_viterbi_k7r2_result_t& result) noexcept;

} // namespace uni::simd::kernels

struct uni_simd_viterbi_k7r2_t {
    uni_simd_viterbi_k7r2_config_t config{};
    uni::simd::kernels::ViterbiState initial_state{};
    uni::simd::kernels::ViterbiState state{};
    uni::simd::kernels::ViterbiRun run{};
    uni_simd_backend_e backend{UNI_SIMD_BACKEND_GENERIC};
};
