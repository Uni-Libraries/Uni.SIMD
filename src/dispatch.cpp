#include <uni/simd/dispatch.hpp>

#include "detail/simd_avx2.h"
#include "detail/simd_avx2fma.h"
#include "detail/simd_avx512bw.h"
#include "detail/simd_avx512f.h"
#include "detail/simd_generic.h"
#include "detail/simd_sse2.h"

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
        return false;
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
    context.backends_.fill(Backend::generic);

#if UNI_SIMD_HAVE_SSE2
    if (caps.sse2 && allows(requested, Backend::sse2)) {
        context.invert_lsb_ = &kernels::Invert1_sse2;
        context.invert_bytes_ = &kernels::Invert8_sse2;
        context.pack_lsb_ = &kernels::Pack8_LSB_sse2;
        context.unpack_lsb_ = &kernels::Unpack8_LSB_sse2;
        context.quantize_ = &kernels::MapQPSK_CF32_U8_sse2;
        context.magnitude_squared_ = &kernels::PowerSpectrumCF32F32_sse2;
        context.psd_ = &kernels::PowerSpectralDensityCF32F32_sse2;
        for (const auto kernel : {Kernel::invert_lsb, Kernel::invert_bytes, Kernel::pack_bits_lsb, Kernel::unpack_bits_lsb,
                                  Kernel::quantize_interleaved_cf32_u8, Kernel::magnitude_squared_cf32}) {
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
        context.backends_.fill(Backend::avx2);
    }
#endif

#if UNI_SIMD_HAVE_AVX2_FMA
    if (caps.avx2 && caps.fma && allows(requested, Backend::avx2_fma)) {
        context.symmetric_dot_ = &kernels::DotProdSymmetricCF32Real_avx2fma;
        context.backends_[static_cast<std::size_t>(Kernel::dot_symmetric_cf32_f32)] = Backend::avx2_fma;
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
    if (!std::isfinite(parameters.scale) || !std::isfinite(parameters.offset)) {
        return Result::invalid_argument;
    }
    if (src.size() > std::numeric_limits<std::size_t>::max() / 2U || dst.size() < src.size() * 2U) {
        return Result::invalid_size;
    }
    if (overlaps(dst, src)) {
        return Result::overlapping_buffers;
    }
    if (src.empty()) {
        return Result::success;
    }
    if (parameters.offset == 128.0f) {
        quantize_(dst.data(), src.data(), src.size(), -parameters.scale);
        return Result::success;
    }
    for (std::size_t i = 0; i < src.size(); ++i) {
        dst[2U * i] = quantize_one(src[i].real(), parameters);
        dst[2U * i + 1U] = quantize_one(src[i].imag(), parameters);
    }
    return Result::success;
}

Result Context::magnitude_squared(const std::span<float> dst, const std::span<const std::complex<float>> src,
                                  const float normalization_factor) const noexcept {
    if (!std::isfinite(normalization_factor) || normalization_factor <= 0.0f) {
        return Result::invalid_argument;
    }
    if (dst.size() < src.size()) {
        return Result::invalid_size;
    }
    if (overlaps(dst, src)) {
        return Result::overlapping_buffers;
    }
    if (!src.empty()) {
        if (math_mode_ == MathMode::deterministic) {
            const float scale = 1.0f / (normalization_factor * normalization_factor);
            for (std::size_t i = 0; i < src.size(); ++i) {
                const float re = src[i].real();
                const float im = src[i].imag();
                dst[i] = (re * re + im * im) * scale;
            }
            return Result::success;
        }
        const float inverse = 1.0f / normalization_factor;
        const float scale = inverse * inverse;
        if (std::isfinite(scale) && scale != 0.0f) {
            magnitude_squared_(dst.data(), src.data(), src.size(), normalization_factor);
        } else {
            const double normalization = normalization_factor;
            for (std::size_t i = 0; i < src.size(); ++i) {
                const double re = static_cast<double>(src[i].real()) / normalization;
                const double im = static_cast<double>(src[i].imag()) / normalization;
                dst[i] = static_cast<float>(re * re + im * im);
            }
        }
    }
    return Result::success;
}

Result Context::power_spectral_density(const std::span<float> dst, const std::span<const std::complex<float>> src,
                                       const float normalization_factor, const float rbw_hz) const noexcept {
    if (!std::isfinite(normalization_factor) || normalization_factor <= 0.0f || !std::isfinite(rbw_hz) || rbw_hz <= 0.0f) {
        return Result::invalid_argument;
    }
    if (dst.size() < src.size()) {
        return Result::invalid_size;
    }
    if (overlaps(dst, src)) {
        return Result::overlapping_buffers;
    }
    if (!src.empty()) {
        if (math_mode_ == MathMode::deterministic) {
            const float inverse_normalization_squared = 1.0f / (normalization_factor * normalization_factor);
            const float scale = inverse_normalization_squared / rbw_hz;
            for (std::size_t i = 0; i < src.size(); ++i) {
                const float re = src[i].real();
                const float im = src[i].imag();
                dst[i] = (re * re + im * im) * scale;
            }
            return Result::success;
        }
        const float inverse = 1.0f / normalization_factor;
        const float scale = (inverse * inverse) / rbw_hz;
        if (std::isfinite(scale) && scale != 0.0f) {
            psd_(dst.data(), src.data(), src.size(), normalization_factor, rbw_hz);
        } else {
            const double normalization = normalization_factor;
            const double rbw = rbw_hz;
            for (std::size_t i = 0; i < src.size(); ++i) {
                const double re = static_cast<double>(src[i].real()) / normalization;
                const double im = static_cast<double>(src[i].imag()) / normalization;
                dst[i] = static_cast<float>((re * re + im * im) / rbw);
            }
        }
    }
    return Result::success;
}

Result Context::dot_cf32_f32(std::complex<float>& dst, const std::span<const std::complex<float>> src,
                             const std::span<const float> taps) const noexcept {
    if (src.size() != taps.size()) {
        dst = {};
        return Result::invalid_size;
    }
    dst = src.empty() ? std::complex<float>{} : dot_(src.data(), taps.data(), src.size());
    return Result::success;
}

Result Context::dot_symmetric_cf32_f32(std::complex<float>& dst, const std::span<const std::complex<float>> src,
                                       const std::span<const float> tap_pairs, const float center_tap) const noexcept {
    if (tap_pairs.size() > (std::numeric_limits<std::size_t>::max() - 1U) / 2U || src.size() != tap_pairs.size() * 2U + 1U) {
        dst = {};
        return Result::invalid_size;
    }
    dst = symmetric_dot_(src.data(), tap_pairs.data(), tap_pairs.size(), center_tap);
    return Result::success;
}

std::complex<float> Context::dot_cf32_f32_unchecked(const std::complex<float>* src, const float* taps, const std::size_t count) const noexcept {
    return count == 0U ? std::complex<float>{} : dot_(src, taps, count);
}

std::complex<float> Context::dot_symmetric_cf32_f32_unchecked(const std::complex<float>* src, const float* tap_pairs, const std::size_t pair_count,
                                                              const float center_tap) const noexcept {
    return symmetric_dot_(src, tap_pairs, pair_count, center_tap);
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
