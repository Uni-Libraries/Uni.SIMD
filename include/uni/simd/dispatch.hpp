#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>

#include <uni/simd/capabilities.hpp>
#include <uni/simd/export.hpp>
#include <uni/simd/result.hpp>

namespace uni::simd {

enum class Backend : std::uint8_t {
    automatic,
    generic,
    sse2,
    avx2,
    avx2_fma,
    avx512,
    neon,
};

enum class MathMode : std::uint8_t {
    // Allows backend-specific reassociation and FMA where available.
    fast,
    // Uses the generic reference backend for reproducible operation ordering.
    deterministic,
};

enum class Kernel : std::uint8_t {
    invert_lsb,
    invert_bytes,
    pack_bits_lsb,
    pack_bits_msb,
    unpack_bits_lsb,
    unpack_bits_msb,
    quantize_interleaved_cf32_u8,
    magnitude_squared_cf32,
    dot_cf32_f32,
    dot_symmetric_cf32_f32,
    count,
};

struct ContextOptions {
    Backend backend = Backend::automatic;
    MathMode math_mode = MathMode::fast;
    bool prefer_energy_efficiency = false;
};

struct QuantizeParameters {
    // Each component maps to round(offset + scale * value), saturated to [0, 255]. NaN maps to the saturated offset.
    float scale = 1.0f;
    float offset = 128.0f;
};

class UNI_SIMD_API Context final {
public:
    // Lengths are element counts. Empty ranges are successful no-ops and no alignment is required.
    // Partial overlap is rejected except by copy, which has memmove semantics. Exact in-place operation is also supported by invert kernels.
    [[nodiscard]] Backend kernel_backend(Kernel kernel) const noexcept;
    [[nodiscard]] MathMode math_mode() const noexcept { return math_mode_; }

    [[nodiscard]] Result copy(std::span<std::uint8_t> dst, std::span<const std::uint8_t> src) const noexcept;
    [[nodiscard]] Result invert_lsb(std::span<std::uint8_t> dst, std::span<const std::uint8_t> src) const noexcept;
    [[nodiscard]] Result invert_bytes(std::span<std::uint8_t> dst, std::span<const std::uint8_t> src) const noexcept;
    [[nodiscard]] Result pack_bits_lsb(std::span<std::uint8_t> dst, std::span<const std::uint8_t> src_bits) const noexcept;
    [[nodiscard]] Result pack_bits_msb(std::span<std::uint8_t> dst, std::span<const std::uint8_t> src_bits) const noexcept;
    [[nodiscard]] Result unpack_bits_lsb(std::span<std::uint8_t> dst_bits, std::span<const std::uint8_t> src) const noexcept;
    [[nodiscard]] Result unpack_bits_msb(std::span<std::uint8_t> dst_bits, std::span<const std::uint8_t> src) const noexcept;
    [[nodiscard]] Result quantize_interleaved_cf32_u8(std::span<std::uint8_t> dst, std::span<const std::complex<float>> src,
                                                     QuantizeParameters parameters = {}) const noexcept;
    [[nodiscard]] Result magnitude_squared(std::span<float> dst, std::span<const std::complex<float>> src,
                                           float normalization_factor = 1.0f) const noexcept;
    [[nodiscard]] Result power_spectral_density(std::span<float> dst, std::span<const std::complex<float>> src, float normalization_factor,
                                                float rbw_hz) const noexcept;
    [[nodiscard]] Result dot_cf32_f32(std::complex<float>& dst, std::span<const std::complex<float>> src,
                                     std::span<const float> taps) const noexcept;
    [[nodiscard]] Result dot_symmetric_cf32_f32(std::complex<float>& dst, std::span<const std::complex<float>> src,
                                               std::span<const float> tap_pairs, float center_tap) const noexcept;

    [[nodiscard]] std::complex<float> dot_cf32_f32_unchecked(const std::complex<float>* src, const float* taps,
                                                            std::size_t count) const noexcept;
    [[nodiscard]] std::complex<float> dot_symmetric_cf32_f32_unchecked(const std::complex<float>* src, const float* tap_pairs,
                                                                      std::size_t pair_count, float center_tap) const noexcept;

private:
    using ByteFn = void (*)(void*, const void*, std::size_t);
    using QuantizeFn = void (*)(void*, const void*, std::size_t, float);
    using PowerFn = void (*)(float*, const std::complex<float>*, std::size_t, float) noexcept;
    using PsdFn = void (*)(float*, const std::complex<float>*, std::size_t, float, float) noexcept;
    using DotFn = std::complex<float> (*)(const void*, const float*, std::size_t) noexcept;
    using SymmetricDotFn = std::complex<float> (*)(const void*, const float*, std::size_t, float) noexcept;

    ByteFn invert_lsb_ = nullptr;
    ByteFn invert_bytes_ = nullptr;
    ByteFn pack_lsb_ = nullptr;
    ByteFn pack_msb_ = nullptr;
    ByteFn unpack_lsb_ = nullptr;
    ByteFn unpack_msb_ = nullptr;
    QuantizeFn quantize_ = nullptr;
    PowerFn magnitude_squared_ = nullptr;
    PsdFn psd_ = nullptr;
    DotFn dot_ = nullptr;
    SymmetricDotFn symmetric_dot_ = nullptr;
    std::array<Backend, static_cast<std::size_t>(Kernel::count)> backends_{};
    MathMode math_mode_ = MathMode::fast;

    friend std::expected<Context, Result> create_context(ContextOptions options) noexcept;
};

[[nodiscard]] UNI_SIMD_API std::expected<Context, Result> create_context(ContextOptions options = {}) noexcept;
[[nodiscard]] UNI_SIMD_API const Context& default_context() noexcept;
[[nodiscard]] UNI_SIMD_API std::string_view backend_name(Backend backend) noexcept;

} // namespace uni::simd
