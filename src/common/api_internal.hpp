#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string_view>

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
    fast,
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
    power_spectral_density_cf32,
    dot_cf32_f32,
    dot_symmetric_cf32_f32,
    count,
};

enum class Result : std::uint8_t {
    success,
    invalid_argument,
    invalid_size,
    overlapping_buffers,
    unsupported_backend,
    out_of_memory,
};

[[nodiscard]] constexpr bool succeeded(const Result result) noexcept { return result == Result::success; }

struct Capabilities {
    bool sse2 = false;
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool avx512bw = false;
    bool neon = false;
};

[[nodiscard]] const Capabilities& capabilities() noexcept;

inline constexpr std::size_t pfb_channelizer_max_taps = 1025U;
inline constexpr std::size_t pfb_channelizer_max_bins = 32U;
inline constexpr std::size_t pfb_channelizer_max_outputs = 8U;

enum class PfbGridOffset : std::uint8_t {
    integer_bins,
    half_bins,
};

struct PfbChannelizerConfig {
    std::size_t bin_count = 0U;
    std::size_t decimation = 0U;
    PfbGridOffset grid_offset = PfbGridOffset::integer_bins;
    std::span<const float> taps{};
    std::span<const std::int32_t> logical_bins{};
};

struct PfbChannelizerBlock {
    std::span<const float> input{};
    std::array<std::span<float>, pfb_channelizer_max_outputs> outputs{};
};

namespace detail {
struct PfbChannelizerData;
}

class Context;

class PfbChannelizer final {
public:
    PfbChannelizer() noexcept;
    ~PfbChannelizer();
    PfbChannelizer(PfbChannelizer&&) noexcept;
    PfbChannelizer& operator=(PfbChannelizer&&) noexcept;
    PfbChannelizer(const PfbChannelizer&) = delete;
    PfbChannelizer& operator=(const PfbChannelizer&) = delete;

    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] Backend backend() const noexcept;
    [[nodiscard]] std::size_t bin_count() const noexcept;
    [[nodiscard]] std::size_t decimation() const noexcept;
    [[nodiscard]] std::size_t tap_count() const noexcept;
    [[nodiscard]] std::span<const std::int32_t> logical_bins() const noexcept;
    [[nodiscard]] Result reset() noexcept;
    [[nodiscard]] std::expected<std::size_t, Result> output_count(std::size_t input_count) const noexcept;
    [[nodiscard]] std::expected<std::size_t, Result> process(const PfbChannelizerBlock& block) noexcept;

private:
    std::unique_ptr<detail::PfbChannelizerData> data_;
    friend class Context;
};

struct IfftSplitComplex {
    std::span<float> real{};
    std::span<float> imag{};
    std::size_t transform_count = 1U;
    std::size_t stride = 0U;
};

class IfftKernel final {
public:
    [[nodiscard]] bool initialized() const noexcept { return function_ != nullptr; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] Backend backend() const noexcept { return backend_; }
    [[nodiscard]] Result execute(IfftSplitComplex values) const noexcept;

private:
    using Function = void (*)(float*, float*, std::size_t) noexcept;
    Function function_ = nullptr;
    std::size_t size_ = 0U;
    Backend backend_ = Backend::generic;
    friend class Context;
};

struct ContextOptions {
    Backend backend = Backend::automatic;
    MathMode math_mode = MathMode::fast;
    bool prefer_energy_efficiency = false;
};

struct QuantizeParameters {
    float scale = 1.0f;
    float offset = 128.0f;
};

class Context final {
public:
    [[nodiscard]] Backend kernel_backend(Kernel kernel) const noexcept;
    [[nodiscard]] MathMode math_mode() const noexcept { return math_mode_; }
    [[nodiscard]] Result copy(std::span<std::uint8_t> dst, std::span<const std::uint8_t> src) const noexcept;
    [[nodiscard]] Result invert_lsb(std::span<std::uint8_t> dst, std::span<const std::uint8_t> src) const noexcept;
    [[nodiscard]] Result invert_bytes(std::span<std::uint8_t> dst, std::span<const std::uint8_t> src) const noexcept;
    [[nodiscard]] Result pack_bits_lsb(std::span<std::uint8_t> dst, std::span<const std::uint8_t> src) const noexcept;
    [[nodiscard]] Result pack_bits_msb(std::span<std::uint8_t> dst, std::span<const std::uint8_t> src) const noexcept;
    [[nodiscard]] Result unpack_bits_lsb(std::span<std::uint8_t> dst, std::span<const std::uint8_t> src) const noexcept;
    [[nodiscard]] Result unpack_bits_msb(std::span<std::uint8_t> dst, std::span<const std::uint8_t> src) const noexcept;
    [[nodiscard]] Result quantize_interleaved_cf32_u8(std::span<std::uint8_t> dst, std::span<const std::complex<float>> src,
                                                     QuantizeParameters parameters = {}) const noexcept;
    [[nodiscard]] Result quantize_interleaved_cf32_u8_raw(std::span<std::uint8_t> dst, const void* src,
                                                         std::size_t count, QuantizeParameters parameters = {}) const noexcept;
    [[nodiscard]] Result magnitude_squared(std::span<float> dst, std::span<const std::complex<float>> src,
                                           float normalization_factor = 1.0f) const noexcept;
    [[nodiscard]] Result magnitude_squared_raw(std::span<float> dst, const void* src, std::size_t count,
                                               float normalization_factor = 1.0f) const noexcept;
    [[nodiscard]] Result power_spectral_density(std::span<float> dst, std::span<const std::complex<float>> src,
                                                float normalization_factor, float rbw_hz) const noexcept;
    [[nodiscard]] Result power_spectral_density_raw(std::span<float> dst, const void* src, std::size_t count,
                                                    float normalization_factor, float rbw_hz) const noexcept;
    [[nodiscard]] Result dot_cf32_f32(std::complex<float>& dst, std::span<const std::complex<float>> src,
                                     std::span<const float> taps) const noexcept;
    [[nodiscard]] Result dot_cf32_f32_raw(std::complex<float>& dst, const void* src, std::size_t count,
                                         std::span<const float> taps) const noexcept;
    [[nodiscard]] Result dot_symmetric_cf32_f32(std::complex<float>& dst, std::span<const std::complex<float>> src,
                                               std::span<const float> taps, float center_tap) const noexcept;
    [[nodiscard]] Result dot_symmetric_cf32_f32_raw(std::complex<float>& dst, const void* src, std::size_t count,
                                                   std::span<const float> taps, float center_tap) const noexcept;
    [[nodiscard]] std::expected<PfbChannelizer, Result> make_pfb_channelizer(const PfbChannelizerConfig& config) const noexcept;
    [[nodiscard]] std::expected<IfftKernel, Result> make_ifft_cf32(std::size_t size) const noexcept;
    [[nodiscard]] std::complex<float> dot_cf32_f32_unchecked(const std::complex<float>* src, const float* taps,
                                                            std::size_t count) const noexcept;
    [[nodiscard]] std::complex<float> dot_symmetric_cf32_f32_unchecked(const std::complex<float>* src,
                                                                      const float* taps, std::size_t pair_count,
                                                                      float center_tap) const noexcept;

private:
    using ByteFn = void (*)(void*, const void*, std::size_t);
    using QuantizeFn = void (*)(void*, const void*, std::size_t, float);
    using PowerFn = void (*)(float*, const void*, std::size_t, float) noexcept;
    using PsdFn = void (*)(float*, const void*, std::size_t, float, float) noexcept;
    using DotFn = std::complex<float> (*)(const void*, const float*, std::size_t) noexcept;
    using SymmetricDotFn = std::complex<float> (*)(const void*, const float*, std::size_t, float) noexcept;
    using PfbChannelizerFn = std::size_t (*)(detail::PfbChannelizerData&, const PfbChannelizerBlock&) noexcept;
    using PfbChannelizerSupportFn = bool (*)(const detail::PfbChannelizerData&) noexcept;
    using IfftFn = void (*)(float*, float*, std::size_t) noexcept;
    using IfftSupportFn = bool (*)(std::size_t) noexcept;

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
    PfbChannelizerFn pfb_channelizer_ = nullptr;
    PfbChannelizerSupportFn pfb_channelizer_support_ = nullptr;
    Backend pfb_channelizer_backend_ = Backend::generic;
    IfftFn ifft_ = nullptr;
    IfftSupportFn ifft_support_ = nullptr;
    Backend ifft_backend_ = Backend::generic;
    std::array<Backend, static_cast<std::size_t>(Kernel::count)> backends_{};
    MathMode math_mode_ = MathMode::fast;
    friend std::expected<Context, Result> create_context(ContextOptions options) noexcept;
};

[[nodiscard]] std::expected<Context, Result> create_context(ContextOptions options = {}) noexcept;
[[nodiscard]] const Context& default_context() noexcept;
[[nodiscard]] std::string_view backend_name(Backend backend) noexcept;

} // namespace uni::simd
