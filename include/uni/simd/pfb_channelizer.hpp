#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include <uni/simd/export.hpp>
#include <uni/simd/result.hpp>

namespace uni::simd {

inline constexpr std::size_t pfb_channelizer_max_taps = 1025U;
inline constexpr std::size_t pfb_channelizer_max_bins = 32U;
inline constexpr std::size_t pfb_channelizer_max_outputs = 8U;
inline constexpr std::size_t pfb_channelizer_max_history = 2048U;
inline constexpr std::size_t pfb_channelizer_max_phase_period = 64U;

enum class PfbGridOffset : std::uint8_t {
    integer_bins,
    half_bins,
};

struct PfbChannelizerPlanConfig {
    std::size_t bin_count = 0U;
    std::size_t decimation = 0U;
    PfbGridOffset grid_offset = PfbGridOffset::integer_bins;
    std::span<const float> taps{};
    std::span<const std::int32_t> logical_bins{};
};

class PfbChannelizerPlan;
class PfbChannelizerState;
struct PfbChannelizerBlockView;

namespace detail {
struct PfbChannelizerAccess;
}

class alignas(64) PfbChannelizerPlan final {
public:
    [[nodiscard]] bool initialized() const noexcept { return signature_ != 0U; }
    [[nodiscard]] std::size_t bin_count() const noexcept { return bin_count_; }
    [[nodiscard]] std::size_t decimation() const noexcept { return decimation_; }
    [[nodiscard]] std::size_t tap_count() const noexcept { return tap_count_; }
    [[nodiscard]] std::size_t selected_output_count() const noexcept { return selected_count_; }
    [[nodiscard]] PfbGridOffset grid_offset() const noexcept { return grid_offset_; }
    [[nodiscard]] std::span<const std::int32_t> logical_bins() const noexcept {
        return {logical_bins_.data(), selected_count_};
    }

private:
    // ceil(1025 / 32) * 32 is the largest padded coefficient set for the supported M values.
    static constexpr std::size_t max_padded_coefficients_ = 1056U;

    std::uint64_t signature_ = 0U;
    std::uint16_t bin_count_ = 0U;
    std::uint16_t decimation_ = 0U;
    std::uint16_t tap_count_ = 0U;
    std::uint16_t row_count_ = 0U;
    std::uint16_t history_size_ = 0U;
    std::uint8_t selected_count_ = 0U;
    std::uint8_t phase_period_ = 1U;
    PfbGridOffset grid_offset_ = PfbGridOffset::integer_bins;
    std::array<std::int32_t, pfb_channelizer_max_outputs> logical_bins_{};
    alignas(64) std::array<float, max_padded_coefficients_> coefficients_{};
    alignas(64) std::array<float, max_padded_coefficients_> reversed_coefficients_{};
    alignas(64) std::array<float, pfb_channelizer_max_bins> branch_rotation_re_{};
    alignas(64) std::array<float, pfb_channelizer_max_bins> branch_rotation_im_{};
    alignas(64) std::array<float, pfb_channelizer_max_bins / 2U> fft_twiddle_re_{};
    alignas(64) std::array<float, pfb_channelizer_max_bins / 2U> fft_twiddle_im_{};
    std::array<std::uint8_t, pfb_channelizer_max_bins> bit_reverse_{};
    alignas(64) std::array<float, pfb_channelizer_max_outputs * pfb_channelizer_max_phase_period> post_phase_re_{};
    alignas(64) std::array<float, pfb_channelizer_max_outputs * pfb_channelizer_max_phase_period> post_phase_im_{};

    friend struct detail::PfbChannelizerAccess;
    friend Result make_pfb_channelizer_plan(PfbChannelizerPlan&, const PfbChannelizerPlanConfig&) noexcept;
    friend Result reset_pfb_channelizer_state(PfbChannelizerState&, const PfbChannelizerPlan&) noexcept;
};

class alignas(64) PfbChannelizerState final {
public:
    [[nodiscard]] bool initialized() const noexcept { return plan_signature_ != 0U; }

private:
    std::uint64_t plan_signature_ = 0U;
    std::uint16_t cursor_ = 0U;
    std::uint16_t decimation_phase_ = 0U;
    std::uint8_t post_phase_ = 0U;
    alignas(64) std::array<float, pfb_channelizer_max_history * 2U> history_i_{};
    alignas(64) std::array<float, pfb_channelizer_max_history * 2U> history_q_{};

    friend struct detail::PfbChannelizerAccess;
    friend Result reset_pfb_channelizer_state(PfbChannelizerState&, const PfbChannelizerPlan&) noexcept;
};

struct PfbChannelizerBlockView {
    std::span<const std::complex<float>> input{};
    std::array<std::span<std::complex<float>>, pfb_channelizer_max_outputs> outputs{};
};

// Plan construction and reset are cold operations. Both validate completely before modifying the destination.
[[nodiscard]] UNI_SIMD_API Result make_pfb_channelizer_plan(PfbChannelizerPlan& destination,
                                                            const PfbChannelizerPlanConfig& config) noexcept;
[[nodiscard]] UNI_SIMD_API Result reset_pfb_channelizer_state(PfbChannelizerState& state,
                                                              const PfbChannelizerPlan& plan) noexcept;
[[nodiscard]] UNI_SIMD_API std::expected<std::size_t, Result>
pfb_channelizer_output_count(const PfbChannelizerPlan& plan, const PfbChannelizerState& state,
                             std::size_t input_count) noexcept;

} // namespace uni::simd
