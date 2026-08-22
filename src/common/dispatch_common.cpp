#include "common/api_internal.hpp"

#include "detail/simd_avx2.h"
#include "detail/simd_avx2fma.h"
#include "detail/simd_avx512bw.h"
#include "detail/simd_avx512f.h"
#include "detail/simd_generic.h"
#include "detail/simd_sse2.h"
#include "ifft_cf32/ifft_cf32_internal.hpp"
#include "pfb_channelizer/pfb_channelizer_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <cstdint>

#ifndef UNI_SIMD_HAVE_SSE2
#define UNI_SIMD_HAVE_SSE2 0
#endif
#ifndef UNI_SIMD_HAVE_AVX2
#define UNI_SIMD_HAVE_AVX2 0
#endif
#ifndef UNI_SIMD_HAVE_AVX2_FMA
#define UNI_SIMD_HAVE_AVX2_FMA 0
#endif
#ifndef UNI_SIMD_HAVE_AVX512F
#define UNI_SIMD_HAVE_AVX512F 0
#endif
#ifndef UNI_SIMD_HAVE_AVX512BW
#define UNI_SIMD_HAVE_AVX512BW 0
#endif
#ifndef UNI_SIMD_HAVE_NEON
#define UNI_SIMD_HAVE_NEON 0
#endif

namespace uni::simd {
namespace {

namespace kernels = uni::simd::detail;

template <typename Left, typename Right>
[[nodiscard]] bool overlaps(const std::span<Left> left, const std::span<Right> right) noexcept {
    if (left.empty() || right.empty()) {
        return false;
    }
    const auto left_begin = reinterpret_cast<std::uintptr_t>(left.data());
    const auto right_begin = reinterpret_cast<std::uintptr_t>(right.data());
    const auto left_end = left_begin + left.size_bytes();
    const auto right_end = right_begin + right.size_bytes();
    return left_begin < right_end && right_begin < left_end;
}

[[nodiscard]] bool overlaps_bytes(const void* const left, const std::size_t left_size,
                                  const void* const right, const std::size_t right_size) noexcept {
    if (left_size == 0U || right_size == 0U) {
        return false;
    }
    const auto left_begin = reinterpret_cast<std::uintptr_t>(left);
    const auto right_begin = reinterpret_cast<std::uintptr_t>(right);
    return left_begin <= right_begin ? right_begin - left_begin < left_size
                                     : left_begin - right_begin < right_size;
}

[[nodiscard]] bool backend_is_available(const Backend backend, const Capabilities& caps) noexcept {
    switch (backend) {
    case Backend::automatic:
    case Backend::generic:
        return true;
    case Backend::sse2:
        return UNI_SIMD_HAVE_SSE2 && caps.sse2;
    case Backend::avx2:
        return UNI_SIMD_HAVE_AVX2 && caps.avx2;
    case Backend::avx2_fma:
        return UNI_SIMD_HAVE_AVX2_FMA && caps.avx2 && caps.fma;
    case Backend::avx512:
        return UNI_SIMD_HAVE_AVX512F && caps.avx512f;
    case Backend::neon:
        return UNI_SIMD_HAVE_NEON && caps.neon;
    }
    return false;
}

[[nodiscard]] std::uint8_t backend_rank(const Backend backend) noexcept {
    switch (backend) {
    case Backend::generic:
        return 0U;
    case Backend::sse2:
        return 1U;
    case Backend::avx2:
        return 2U;
    case Backend::avx2_fma:
        return 3U;
    case Backend::avx512:
        return 4U;
    case Backend::neon:
        return 1U;
    case Backend::automatic:
        return std::numeric_limits<std::uint8_t>::max();
    }
    return 0U;
}

[[nodiscard]] bool allows(const Backend requested, const Backend candidate) noexcept {
    return requested == Backend::automatic || backend_rank(candidate) <= backend_rank(requested);
}

[[nodiscard]] std::uint8_t quantize_one(const float value, const QuantizeParameters parameters) noexcept {
    const float mapped = parameters.offset + parameters.scale * value;
    if (std::isnan(mapped)) {
        return static_cast<std::uint8_t>(std::lrint(std::clamp(parameters.offset, 0.0f, 255.0f)));
    }
    if (mapped <= 0.0f) {
        return 0U;
    }
    if (mapped >= 255.0f) {
        return 255U;
    }
    return static_cast<std::uint8_t>(std::lrint(mapped));
}

} // namespace

std::expected<Context, Result> create_context(const ContextOptions options) noexcept {
    const auto& caps = capabilities();
    Backend requested = options.backend;
    if (options.math_mode == MathMode::deterministic) {
        if (requested != Backend::automatic && requested != Backend::generic) {
            return std::unexpected(Result::invalid_argument);
        }
        requested = Backend::generic;
    }
    if (requested != Backend::automatic && !backend_is_available(requested, caps)) {
        return std::unexpected(Result::unsupported_backend);
    }

    Context context{};
    context.math_mode_ = options.math_mode;
    context.invert_lsb_ = &kernels::Invert1_generic;
    context.invert_bytes_ = &kernels::Invert8_generic;
    context.pack_lsb_ = &kernels::Pack8_LSB_generic;
    context.pack_msb_ = &kernels::Pack8_MSB_generic;
    context.unpack_lsb_ = &kernels::Unpack8_LSB_generic;
    context.unpack_msb_ = &kernels::Unpack8_MSB_generic;
    context.quantize_ = &kernels::MapQPSK_CF32_U8_generic;
    context.magnitude_squared_ = &kernels::PowerSpectrumCF32F32_generic;
    context.psd_ = &kernels::PowerSpectralDensityCF32F32_generic;
    context.dot_ = &kernels::DotProdCF32Real_generic;
    context.symmetric_dot_ = &kernels::DotProdSymmetricCF32Real_generic;
    context.pfb_channelizer_ = &kernels::PfbChannelizer_generic;
    context.pfb_channelizer_support_ = &kernels::PfbChannelizer_supports_all;
    context.ifft_ = &kernels::Ifft_generic;
    context.ifft_support_ = &kernels::Ifft_supports_all;
    context.backends_.fill(Backend::generic);

#if UNI_SIMD_HAVE_SSE2
    if (caps.sse2 && allows(requested, Backend::sse2)) {
        context.invert_lsb_ = &kernels::Invert1_sse2;
        context.invert_bytes_ = &kernels::Invert8_sse2;
        context.pack_lsb_ = &kernels::Pack8_LSB_sse2;
        context.pack_msb_ = &kernels::Pack8_MSB_sse2;
        context.unpack_lsb_ = &kernels::Unpack8_LSB_sse2;
        context.unpack_msb_ = &kernels::Unpack8_MSB_sse2;
        context.quantize_ = &kernels::MapQPSK_CF32_U8_sse2;
        context.magnitude_squared_ = &kernels::PowerSpectrumCF32F32_sse2;
        context.psd_ = &kernels::PowerSpectralDensityCF32F32_sse2;
        for (const auto kernel : {Kernel::invert_lsb, Kernel::invert_bytes, Kernel::pack_bits_lsb, Kernel::pack_bits_msb,
                                  Kernel::unpack_bits_lsb, Kernel::unpack_bits_msb, Kernel::quantize_interleaved_cf32_u8,
                                  Kernel::magnitude_squared_cf32, Kernel::power_spectral_density_cf32}) {
            context.backends_[static_cast<std::size_t>(kernel)] = Backend::sse2;
        }
    }
#endif

#if UNI_SIMD_HAVE_AVX2
    if (caps.avx2 && allows(requested, Backend::avx2)) {
        context.invert_lsb_ = &kernels::Invert1_avx2;
        context.invert_bytes_ = &kernels::Invert8_avx2;
        context.pack_lsb_ = &kernels::Pack8_LSB_avx2;
        context.pack_msb_ = &kernels::Pack8_MSB_avx2;
        context.unpack_lsb_ = &kernels::Unpack8_LSB_avx2;
        context.unpack_msb_ = &kernels::Unpack8_MSB_avx2;
        context.quantize_ = &kernels::MapQPSK_CF32_U8_avx2;
        context.magnitude_squared_ = &kernels::PowerSpectrumCF32F32_avx2;
        context.psd_ = &kernels::PowerSpectralDensityCF32F32_avx2;
        context.dot_ = &kernels::DotProdCF32Real_avx2;
        context.symmetric_dot_ = &kernels::DotProdSymmetricCF32Real_avx2;
        for (const auto kernel : {Kernel::invert_lsb, Kernel::invert_bytes, Kernel::pack_bits_lsb, Kernel::pack_bits_msb,
                                  Kernel::unpack_bits_lsb, Kernel::unpack_bits_msb, Kernel::quantize_interleaved_cf32_u8,
                                   Kernel::magnitude_squared_cf32, Kernel::power_spectral_density_cf32,
                                   Kernel::dot_cf32_f32, Kernel::dot_symmetric_cf32_f32}) {
            context.backends_[static_cast<std::size_t>(kernel)] = Backend::avx2;
        }
    }
#endif

#if UNI_SIMD_HAVE_AVX2_FMA
    if (caps.avx2 && caps.fma && allows(requested, Backend::avx2_fma)) {
        context.dot_ = &kernels::DotProdCF32Real_avx2fma;
        context.symmetric_dot_ = &kernels::DotProdSymmetricCF32Real_avx2fma;
        context.pfb_channelizer_ = &kernels::PfbChannelizer_avx2fma;
        context.pfb_channelizer_support_ = &kernels::PfbChannelizer_supports_m8_d4;
        context.ifft_ = &kernels::Ifft8_avx2_fma;
        context.ifft_support_ = &kernels::Ifft_supports_8;
        context.ifft_backend_ = Backend::avx2_fma;
        context.pfb_channelizer_backend_ = Backend::avx2_fma;
        context.backends_[static_cast<std::size_t>(Kernel::dot_cf32_f32)] = Backend::avx2_fma;
        context.backends_[static_cast<std::size_t>(Kernel::dot_symmetric_cf32_f32)] = Backend::avx2_fma;
    }
#endif

#if UNI_SIMD_HAVE_NEON
    if (caps.neon && allows(requested, Backend::neon)) {
        context.pfb_channelizer_ = &kernels::PfbChannelizer_neon;
        context.pfb_channelizer_support_ = &kernels::PfbChannelizer_supports_m8_d4;
        context.pfb_channelizer_backend_ = Backend::neon;
    }
#endif

#if UNI_SIMD_HAVE_AVX512F
    const bool allow_avx512 = requested != Backend::automatic || !options.prefer_energy_efficiency;
    if (allow_avx512 && caps.avx512f && allows(requested, Backend::avx512)) {
        context.invert_lsb_ = &kernels::Invert1_avx512f;
        context.invert_bytes_ = &kernels::Invert8_avx512f;
        context.backends_[static_cast<std::size_t>(Kernel::invert_lsb)] = Backend::avx512;
        context.backends_[static_cast<std::size_t>(Kernel::invert_bytes)] = Backend::avx512;
    }
#endif
#if UNI_SIMD_HAVE_AVX512BW
    if (allow_avx512 && caps.avx512bw && allows(requested, Backend::avx512)) {
        context.pack_lsb_ = &kernels::Pack8_LSB_avx512bw;
        context.unpack_lsb_ = &kernels::Unpack8_LSB_avx512bw;
        context.backends_[static_cast<std::size_t>(Kernel::pack_bits_lsb)] = Backend::avx512;
        context.backends_[static_cast<std::size_t>(Kernel::unpack_bits_lsb)] = Backend::avx512;
    }
#endif

    return context;
}

const Context& default_context() noexcept {
    static const Context context = *create_context();
    return context;
}

Backend Context::kernel_backend(const Kernel kernel) const noexcept {
    const auto index = static_cast<std::size_t>(kernel);
    return index < backends_.size() ? backends_[index] : Backend::generic;
}

Result Context::copy(const std::span<std::uint8_t> dst, const std::span<const std::uint8_t> src) const noexcept {
    if (dst.size() < src.size()) {
        return Result::invalid_size;
    }
    if (!src.empty()) {
        std::memmove(dst.data(), src.data(), src.size());
    }
    return Result::success;
}

Result Context::invert_lsb(const std::span<std::uint8_t> dst, const std::span<const std::uint8_t> src) const noexcept {
    if (dst.size() < src.size()) {
        return Result::invalid_size;
    }
    if (overlaps(dst, src) && dst.data() != src.data()) {
        return Result::overlapping_buffers;
    }
    if (!src.empty()) {
        invert_lsb_(dst.data(), src.data(), src.size());
    }
    return Result::success;
}

Result Context::invert_bytes(const std::span<std::uint8_t> dst, const std::span<const std::uint8_t> src) const noexcept {
    if (dst.size() < src.size()) {
        return Result::invalid_size;
    }
    if (overlaps(dst, src) && dst.data() != src.data()) {
        return Result::overlapping_buffers;
    }
    if (!src.empty()) {
        invert_bytes_(dst.data(), src.data(), src.size());
    }
    return Result::success;
}

Result Context::pack_bits_lsb(const std::span<std::uint8_t> dst, const std::span<const std::uint8_t> src_bits) const noexcept {
    if (src_bits.size() % 8U != 0U || dst.size() < src_bits.size() / 8U) {
        return Result::invalid_size;
    }
    if (overlaps(dst, src_bits)) {
        return Result::overlapping_buffers;
    }
    if (!src_bits.empty()) {
        pack_lsb_(dst.data(), src_bits.data(), src_bits.size() / 8U);
    }
    return Result::success;
}

Result Context::pack_bits_msb(const std::span<std::uint8_t> dst, const std::span<const std::uint8_t> src_bits) const noexcept {
    if (src_bits.size() % 8U != 0U || dst.size() < src_bits.size() / 8U) {
        return Result::invalid_size;
    }
    if (overlaps(dst, src_bits)) {
        return Result::overlapping_buffers;
    }
    if (!src_bits.empty()) {
        pack_msb_(dst.data(), src_bits.data(), src_bits.size() / 8U);
    }
    return Result::success;
}

Result Context::unpack_bits_lsb(const std::span<std::uint8_t> dst_bits, const std::span<const std::uint8_t> src) const noexcept {
    if (src.size() > std::numeric_limits<std::size_t>::max() / 8U || dst_bits.size() < src.size() * 8U) {
        return Result::invalid_size;
    }
    if (overlaps(dst_bits, src)) {
        return Result::overlapping_buffers;
    }
    if (!src.empty()) {
        unpack_lsb_(dst_bits.data(), src.data(), src.size());
    }
    return Result::success;
}

Result Context::unpack_bits_msb(const std::span<std::uint8_t> dst_bits, const std::span<const std::uint8_t> src) const noexcept {
    if (src.size() > std::numeric_limits<std::size_t>::max() / 8U || dst_bits.size() < src.size() * 8U) {
        return Result::invalid_size;
    }
    if (overlaps(dst_bits, src)) {
        return Result::overlapping_buffers;
    }
    if (!src.empty()) {
        unpack_msb_(dst_bits.data(), src.data(), src.size());
    }
    return Result::success;
}

Result Context::quantize_interleaved_cf32_u8(const std::span<std::uint8_t> dst, const std::span<const std::complex<float>> src,
                                             const QuantizeParameters parameters) const noexcept {
    return quantize_interleaved_cf32_u8_raw(dst, src.data(), src.size(), parameters);
}

Result Context::quantize_interleaved_cf32_u8_raw(const std::span<std::uint8_t> dst, const void* const src,
                                                 const std::size_t count,
                                                 const QuantizeParameters parameters) const noexcept {
    if (!std::isfinite(parameters.scale) || !std::isfinite(parameters.offset)) {
        return Result::invalid_argument;
    }
    if ((count != 0U && src == nullptr) ||
        count > std::numeric_limits<std::size_t>::max() / (2U * sizeof(float)) || dst.size() < count * 2U) {
        return Result::invalid_size;
    }
    if (overlaps_bytes(dst.data(), dst.size_bytes(), src, count * 2U * sizeof(float))) {
        return Result::overlapping_buffers;
    }
    if (count == 0U) {
        return Result::success;
    }
    if (parameters.offset == 128.0f) {
        quantize_(dst.data(), src, count, -parameters.scale);
        return Result::success;
    }
    const auto* const components = static_cast<const float*>(src);
    for (std::size_t i = 0; i < count; ++i) {
        dst[2U * i] = quantize_one(components[2U * i], parameters);
        dst[2U * i + 1U] = quantize_one(components[2U * i + 1U], parameters);
    }
    return Result::success;
}

Result Context::magnitude_squared(const std::span<float> dst, const std::span<const std::complex<float>> src,
                                  const float normalization_factor) const noexcept {
    return magnitude_squared_raw(dst, src.data(), src.size(), normalization_factor);
}

Result Context::magnitude_squared_raw(const std::span<float> dst, const void* const src,
                                      const std::size_t count, const float normalization_factor) const noexcept {
    if (!std::isfinite(normalization_factor) || normalization_factor <= 0.0f) {
        return Result::invalid_argument;
    }
    if ((count != 0U && src == nullptr) ||
        count > std::numeric_limits<std::size_t>::max() / (2U * sizeof(float)) ||
        dst.size() > std::numeric_limits<std::size_t>::max() / sizeof(float) || dst.size() < count) {
        return Result::invalid_size;
    }
    if (overlaps_bytes(dst.data(), dst.size_bytes(), src, count * 2U * sizeof(float))) {
        return Result::overlapping_buffers;
    }
    if (count != 0U) {
        if (math_mode_ == MathMode::deterministic) {
            kernels::PowerSpectrumCF32F32_deterministic(dst.data(), src, count, normalization_factor);
            return Result::success;
        }
        const auto* const components = static_cast<const float*>(src);
        const float inverse = 1.0f / normalization_factor;
        const float scale = inverse * inverse;
        if (std::isfinite(scale) && scale != 0.0f) {
            magnitude_squared_(dst.data(), src, count, normalization_factor);
        } else {
            const double normalization = normalization_factor;
            for (std::size_t i = 0; i < count; ++i) {
                const double re = static_cast<double>(components[2U * i]) / normalization;
                const double im = static_cast<double>(components[2U * i + 1U]) / normalization;
                dst[i] = static_cast<float>(re * re + im * im);
            }
        }
    }
    return Result::success;
}

Result Context::power_spectral_density(const std::span<float> dst, const std::span<const std::complex<float>> src,
                                       const float normalization_factor, const float rbw_hz) const noexcept {
    return power_spectral_density_raw(dst, src.data(), src.size(), normalization_factor, rbw_hz);
}

Result Context::power_spectral_density_raw(const std::span<float> dst, const void* const src,
                                           const std::size_t count, const float normalization_factor,
                                           const float rbw_hz) const noexcept {
    if (!std::isfinite(normalization_factor) || normalization_factor <= 0.0f || !std::isfinite(rbw_hz) || rbw_hz <= 0.0f) {
        return Result::invalid_argument;
    }
    if ((count != 0U && src == nullptr) ||
        count > std::numeric_limits<std::size_t>::max() / (2U * sizeof(float)) ||
        dst.size() > std::numeric_limits<std::size_t>::max() / sizeof(float) || dst.size() < count) {
        return Result::invalid_size;
    }
    if (overlaps_bytes(dst.data(), dst.size_bytes(), src, count * 2U * sizeof(float))) {
        return Result::overlapping_buffers;
    }
    if (count != 0U) {
        if (math_mode_ == MathMode::deterministic) {
            kernels::PowerSpectralDensityCF32F32_deterministic(
                dst.data(), src, count, normalization_factor, rbw_hz);
            return Result::success;
        }
        const auto* const components = static_cast<const float*>(src);
        const float inverse = 1.0f / normalization_factor;
        const float scale = (inverse * inverse) / rbw_hz;
        if (std::isfinite(scale) && scale != 0.0f) {
            psd_(dst.data(), src, count, normalization_factor, rbw_hz);
        } else {
            const double normalization = normalization_factor;
            const double rbw = rbw_hz;
            for (std::size_t i = 0; i < count; ++i) {
                const double re = static_cast<double>(components[2U * i]) / normalization;
                const double im = static_cast<double>(components[2U * i + 1U]) / normalization;
                dst[i] = static_cast<float>((re * re + im * im) / rbw);
            }
        }
    }
    return Result::success;
}

Result Context::dot_cf32_f32(std::complex<float>& dst, const std::span<const std::complex<float>> src,
                             const std::span<const float> taps) const noexcept {
    return dot_cf32_f32_raw(dst, src.data(), src.size(), taps);
}

Result Context::dot_cf32_f32_raw(std::complex<float>& dst, const void* const src, const std::size_t count,
                                 const std::span<const float> taps) const noexcept {
    if ((count != 0U && src == nullptr) ||
        count > std::numeric_limits<std::size_t>::max() / (2U * sizeof(float)) ||
        count != taps.size()) {
        dst = {};
        return Result::invalid_size;
    }
    dst = count == 0U ? std::complex<float>{} : dot_(src, taps.data(), count);
    return Result::success;
}

Result Context::dot_symmetric_cf32_f32(std::complex<float>& dst, const std::span<const std::complex<float>> src,
                                       const std::span<const float> tap_pairs, const float center_tap) const noexcept {
    return dot_symmetric_cf32_f32_raw(dst, src.data(), src.size(), tap_pairs, center_tap);
}

Result Context::dot_symmetric_cf32_f32_raw(std::complex<float>& dst, const void* const src,
                                           const std::size_t count, const std::span<const float> tap_pairs,
                                           const float center_tap) const noexcept {
    if (src == nullptr || tap_pairs.size() > std::numeric_limits<std::size_t>::max() / 4U ||
        count != tap_pairs.size() * 2U + 1U) {
        dst = {};
        return Result::invalid_size;
    }
    dst = symmetric_dot_(src, tap_pairs.data(), tap_pairs.size(), center_tap);
    return Result::success;
}

std::complex<float> Context::dot_cf32_f32_unchecked(const std::complex<float>* src, const float* taps, const std::size_t count) const noexcept {
    return count == 0U ? std::complex<float>{} : dot_(src, taps, count);
}

std::complex<float> Context::dot_symmetric_cf32_f32_unchecked(const std::complex<float>* src, const float* tap_pairs, const std::size_t pair_count,
                                                              const float center_tap) const noexcept {
    return symmetric_dot_(src, tap_pairs, pair_count, center_tap);
}

std::expected<PfbChannelizer, Result>
Context::make_pfb_channelizer(const PfbChannelizerConfig& config) const noexcept {
    auto data = detail::make_pfb_channelizer_data(
        config, pfb_channelizer_, pfb_channelizer_support_,
        pfb_channelizer_backend_);
    if (!data) {
        return std::unexpected(data.error());
    }
    PfbChannelizer channelizer;
    channelizer.data_ = std::move(*data);
    return channelizer;
}

std::expected<IfftKernel, Result> Context::make_ifft_cf32(const std::size_t size) const noexcept {
    if (!kernels::Ifft_supports_all(size)) {
        return std::unexpected(Result::invalid_argument);
    }
    IfftKernel kernel;
    if (ifft_support_(size)) {
        kernel.function_ = ifft_;
        kernel.backend_ = ifft_backend_;
    } else {
        kernel.function_ = &kernels::Ifft_generic;
        kernel.backend_ = Backend::generic;
    }
    kernel.size_ = size;
    return kernel;
}

std::string_view backend_name(const Backend backend) noexcept {
    switch (backend) {
    case Backend::automatic:
        return "automatic";
    case Backend::generic:
        return "generic";
    case Backend::sse2:
        return "sse2";
    case Backend::avx2:
        return "avx2";
    case Backend::avx2_fma:
        return "avx2-fma";
    case Backend::avx512:
        return "avx512";
    case Backend::neon:
        return "neon";
    }
    return "unknown";
}

} // namespace uni::simd
