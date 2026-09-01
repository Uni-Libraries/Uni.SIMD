#pragma once

#include "common/api_internal.hpp"
#include "uni_simd_typedefs.h"

#include <array>
#include <cstddef>

struct uni_simd_qpsk_costas4_t;

namespace uni::simd::kernels {

using Costas4Process = void (*)(uni_simd_qpsk_costas4_t&, const uni_simd_qpsk_costas4_block_t&) noexcept;

void QpskCostas4_generic(uni_simd_qpsk_costas4_t& kernel,
                         const uni_simd_qpsk_costas4_block_t& block) noexcept;
#if UNI_SIMD_HAVE_AVX2_FMA
void QpskCostas4_avx2(uni_simd_qpsk_costas4_t& kernel,
                      const uni_simd_qpsk_costas4_block_t& block) noexcept;
#endif

void Costas4Normalize(float& phase, float& phase_cos, float& phase_sin) noexcept;
void Costas4SinCos(float delta, float& phase_sin, float& phase_cos) noexcept;

[[nodiscard]] uni_simd_result_e QpskCostas4Initialize(
    uni_simd_qpsk_costas4_t& kernel,
    const uni_simd_qpsk_costas4_config_t& config,
    uni_simd_backend_e requested_backend,
    uni_simd_math_mode_e math_mode) noexcept;
[[nodiscard]] uni_simd_result_e QpskCostas4Reset(uni_simd_qpsk_costas4_t& kernel) noexcept;
[[nodiscard]] uni_simd_result_e QpskCostas4Execute(
    uni_simd_qpsk_costas4_t& kernel,
    const uni_simd_qpsk_costas4_block_t& block,
    uni_simd_qpsk_costas4_state_t* final_state) noexcept;

} // namespace uni::simd::kernels

struct uni_simd_qpsk_costas4_t {
    uni_simd_qpsk_costas4_config_t config{};
    uni_simd_qpsk_costas4_state_t initial_state{};
    uni_simd_qpsk_costas4_state_t state{};
    uni::simd::kernels::Costas4Process process{};
    uni_simd_backend_e backend{UNI_SIMD_BACKEND_GENERIC};
    std::size_t samples_since_normalization{};
};
