#include <uni_simd.h>

#include "common/api_internal.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <utility>

static_assert(static_cast<unsigned>(uni::simd::Backend::automatic) == UNI_SIMD_BACKEND_AUTOMATIC);
static_assert(static_cast<unsigned>(uni::simd::Backend::neon) == UNI_SIMD_BACKEND_AARCH64_NEON);

struct uni_simd_state_t final {
    uni::simd::PfbChannelizer pfb;
};

namespace {

struct Runtime final {
    std::shared_mutex mutex;
    bool initialized = false;
    std::atomic_size_t state_count{0U};
};

[[nodiscard]] Runtime& runtime() noexcept {
    static Runtime value;
    return value;
}

struct ContextCache final {
    std::array<std::optional<uni::simd::Context>, 7U> fast{};
    std::array<uni::simd::Result, 7U> errors{};
    std::optional<uni::simd::Context> energy_efficient;
    std::optional<uni::simd::Context> deterministic;

    ContextCache() noexcept {
        errors.fill(uni::simd::Result::unsupported_backend);
        for (std::size_t index = 0U; index < fast.size(); ++index) {
            const auto created = uni::simd::create_context({.backend = static_cast<uni::simd::Backend>(index)});
            if (created) {
                fast[index] = *created;
                errors[index] = uni::simd::Result::success;
            } else {
                errors[index] = created.error();
            }
        }
        if (auto created = uni::simd::create_context({.prefer_energy_efficiency = true})) {
            energy_efficient = *created;
        }
        if (auto created = uni::simd::create_context({.math_mode = uni::simd::MathMode::deterministic})) {
            deterministic = *created;
        }
    }
};

[[nodiscard]] ContextCache& context_cache() noexcept {
    static ContextCache cache;
    return cache;
}

struct ParsedParams final {
    std::array<bool, UNI_SIMD_PARAM_RESET + 1U> present{};
    uni_simd_backend_e backend = UNI_SIMD_BACKEND_AUTOMATIC;
    uni_simd_math_mode_e math_mode = UNI_SIMD_MATH_FAST;
    bool prefer_energy_efficiency = false;
    float scale = 1.0f;
    float offset = 128.0f;
    float normalization_factor = 1.0f;
    float rbw_hz = 0.0f;
    float center_tap = 0.0f;
    const float* taps = nullptr;
    std::size_t tap_count = 0U;
    std::size_t bin_count = 0U;
    std::size_t decimation = 0U;
    uni_simd_pfb_grid_offset_e grid_offset = UNI_SIMD_PFB_INTEGER_BINS;
    const std::int32_t* logical_bins = nullptr;
    std::size_t logical_bin_count = 0U;
    bool query_output_count = false;
    bool reset = false;
    bool has_rbw_hz = false;
    bool has_taps = false;
    bool has_tap_count = false;
    bool has_center_tap = false;
    bool has_bin_count = false;
    bool has_decimation = false;
    bool has_grid_offset = false;
    bool has_logical_bins = false;
    bool has_logical_bin_count = false;
    uni_simd_param_t* resolved_backend = nullptr;
    uni_simd_param_t* output_count = nullptr;
};

[[nodiscard]] bool is_u32(const uni_simd_param_t& param) noexcept {
    return param.param_type == UNI_SIMD_PARAM_TYPE_U32;
}

[[nodiscard]] bool is_size(const uni_simd_param_t& param) noexcept {
    return param.param_type == UNI_SIMD_PARAM_TYPE_SIZE;
}

[[nodiscard]] bool is_f32(const uni_simd_param_t& param) noexcept {
    return param.param_type == UNI_SIMD_PARAM_TYPE_FLOAT32;
}

[[nodiscard]] bool is_pointer(const uni_simd_param_t& param) noexcept {
    return param.param_type == UNI_SIMD_PARAM_TYPE_CONST_POINTER;
}

[[nodiscard]] uni_simd_result_e parse_params(uni_simd_param_t* const params, const std::size_t count,
                                             ParsedParams& parsed) noexcept {
    if (count != 0U && params == nullptr) {
        return UNI_SIMD_RESULT_INVALID_ARGUMENT;
    }
    for (std::size_t index = 0U; index < count; ++index) {
        auto& param = params[index];
        const auto id = static_cast<std::size_t>(param.param_id);
        if (id == 0U || id >= parsed.present.size() || parsed.present[id]) {
            return UNI_SIMD_RESULT_INVALID_ARGUMENT;
        }
        parsed.present[id] = true;
        switch (param.param_id) {
        case UNI_SIMD_PARAM_BACKEND:
            if (!is_u32(param) || param.value.u32 > UNI_SIMD_BACKEND_AARCH64_NEON) {
                return UNI_SIMD_RESULT_INVALID_ARGUMENT;
            }
            parsed.backend = static_cast<uni_simd_backend_e>(param.value.u32);
            break;
        case UNI_SIMD_PARAM_RESOLVED_BACKEND:
            if (!is_u32(param)) {
                return UNI_SIMD_RESULT_INVALID_ARGUMENT;
            }
            parsed.resolved_backend = &param;
            break;
        case UNI_SIMD_PARAM_MATH_MODE:
            if (!is_u32(param) || param.value.u32 > UNI_SIMD_MATH_DETERMINISTIC) {
                return UNI_SIMD_RESULT_INVALID_ARGUMENT;
            }
            parsed.math_mode = static_cast<uni_simd_math_mode_e>(param.value.u32);
            break;
        case UNI_SIMD_PARAM_PREFER_ENERGY_EFFICIENCY:
            if (!is_u32(param) || param.value.u32 > 1U) {
                return UNI_SIMD_RESULT_INVALID_ARGUMENT;
            }
            parsed.prefer_energy_efficiency = param.value.u32 != 0U;
            break;
        case UNI_SIMD_PARAM_SCALE:
            if (!is_f32(param)) {
                return UNI_SIMD_RESULT_INVALID_ARGUMENT;
            }
            parsed.scale = param.value.f32;
            break;
        case UNI_SIMD_PARAM_OFFSET:
            if (!is_f32(param)) {
                return UNI_SIMD_RESULT_INVALID_ARGUMENT;
            }
            parsed.offset = param.value.f32;
            break;
        case UNI_SIMD_PARAM_NORMALIZATION_FACTOR:
            if (!is_f32(param)) {
                return UNI_SIMD_RESULT_INVALID_ARGUMENT;
            }
            parsed.normalization_factor = param.value.f32;
            break;
        case UNI_SIMD_PARAM_RBW_HZ:
            if (!is_f32(param)) {
                return UNI_SIMD_RESULT_INVALID_ARGUMENT;
            }
            parsed.rbw_hz = param.value.f32;
            parsed.has_rbw_hz = true;
            break;
        case UNI_SIMD_PARAM_TAPS:
            if (!is_pointer(param)) {
                return UNI_SIMD_RESULT_INVALID_ARGUMENT;
            }
            parsed.taps = static_cast<const float*>(param.value.const_pointer);
            parsed.has_taps = true;
            break;
        case UNI_SIMD_PARAM_TAP_COUNT:
            if (!is_size(param)) {
                return UNI_SIMD_RESULT_INVALID_ARGUMENT;
            }
            parsed.tap_count = param.value.size;
            parsed.has_tap_count = true;
            break;
        case UNI_SIMD_PARAM_CENTER_TAP:
            if (!is_f32(param)) {
                return UNI_SIMD_RESULT_INVALID_ARGUMENT;
            }
            parsed.center_tap = param.value.f32;
            parsed.has_center_tap = true;
            break;
        case UNI_SIMD_PARAM_BIN_COUNT:
            if (!is_size(param)) {
                return UNI_SIMD_RESULT_INVALID_ARGUMENT;
            }
            parsed.bin_count = param.value.size;
            parsed.has_bin_count = true;
            break;
        case UNI_SIMD_PARAM_DECIMATION:
            if (!is_size(param)) {
                return UNI_SIMD_RESULT_INVALID_ARGUMENT;
            }
            parsed.decimation = param.value.size;
            parsed.has_decimation = true;
            break;
        case UNI_SIMD_PARAM_GRID_OFFSET:
            if (!is_u32(param) || param.value.u32 > UNI_SIMD_PFB_HALF_BINS) {
                return UNI_SIMD_RESULT_INVALID_ARGUMENT;
            }
            parsed.grid_offset = static_cast<uni_simd_pfb_grid_offset_e>(param.value.u32);
            parsed.has_grid_offset = true;
            break;
        case UNI_SIMD_PARAM_LOGICAL_BINS:
            if (!is_pointer(param)) {
                return UNI_SIMD_RESULT_INVALID_ARGUMENT;
            }
            parsed.logical_bins = static_cast<const std::int32_t*>(param.value.const_pointer);
            parsed.has_logical_bins = true;
            break;
        case UNI_SIMD_PARAM_LOGICAL_BIN_COUNT:
            if (!is_size(param)) {
                return UNI_SIMD_RESULT_INVALID_ARGUMENT;
            }
            parsed.logical_bin_count = param.value.size;
            parsed.has_logical_bin_count = true;
            break;
        case UNI_SIMD_PARAM_OUTPUT_COUNT:
            if (!is_size(param)) {
                return UNI_SIMD_RESULT_INVALID_ARGUMENT;
            }
            parsed.output_count = &param;
            break;
        case UNI_SIMD_PARAM_QUERY_OUTPUT_COUNT:
            if (!is_u32(param) || param.value.u32 > 1U) {
                return UNI_SIMD_RESULT_INVALID_ARGUMENT;
            }
            parsed.query_output_count = param.value.u32 != 0U;
            break;
        case UNI_SIMD_PARAM_RESET:
            if (!is_u32(param) || param.value.u32 > 1U) {
                return UNI_SIMD_RESULT_INVALID_ARGUMENT;
            }
            parsed.reset = param.value.u32 != 0U;
            break;
        case UNI_SIMD_PARAM_UNKNOWN:
            return UNI_SIMD_RESULT_INVALID_ARGUMENT;
        }
    }
    return UNI_SIMD_RESULT_SUCCESS;
}

[[nodiscard]] bool parameters_allowed(const uni_simd_kernel_e kernel,
                                      const ParsedParams& params) noexcept {
    const auto allowed = [kernel](const std::size_t id) noexcept {
        switch (id) {
        case UNI_SIMD_PARAM_BACKEND:
        case UNI_SIMD_PARAM_RESOLVED_BACKEND:
        case UNI_SIMD_PARAM_MATH_MODE:
        case UNI_SIMD_PARAM_PREFER_ENERGY_EFFICIENCY:
            return true;
        default:
            break;
        }
        switch (kernel) {
        case UNI_SIMD_KERNEL_QUANTIZE_CF32_U8:
            return id == UNI_SIMD_PARAM_SCALE || id == UNI_SIMD_PARAM_OFFSET;
        case UNI_SIMD_KERNEL_MAGNITUDE_SQUARED_CF32_F32:
            return id == UNI_SIMD_PARAM_NORMALIZATION_FACTOR;
        case UNI_SIMD_KERNEL_POWER_SPECTRAL_DENSITY_CF32_F32:
            return id == UNI_SIMD_PARAM_NORMALIZATION_FACTOR || id == UNI_SIMD_PARAM_RBW_HZ;
        case UNI_SIMD_KERNEL_DOT_CF32_F32:
            return id == UNI_SIMD_PARAM_TAPS || id == UNI_SIMD_PARAM_TAP_COUNT;
        case UNI_SIMD_KERNEL_DOT_SYMMETRIC_CF32_F32:
            return id == UNI_SIMD_PARAM_TAPS || id == UNI_SIMD_PARAM_TAP_COUNT ||
                   id == UNI_SIMD_PARAM_CENTER_TAP;
        case UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32:
            return id >= UNI_SIMD_PARAM_TAPS && id <= UNI_SIMD_PARAM_RESET &&
                   id != UNI_SIMD_PARAM_CENTER_TAP;
        default:
            return false;
        }
    };
    for (std::size_t id = 1U; id < params.present.size(); ++id) {
        if (params.present[id] && !allowed(id)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool has_pfb_creation_params(const ParsedParams& params) noexcept {
    for (const auto id : {UNI_SIMD_PARAM_BACKEND, UNI_SIMD_PARAM_MATH_MODE,
                          UNI_SIMD_PARAM_PREFER_ENERGY_EFFICIENCY, UNI_SIMD_PARAM_TAPS,
                          UNI_SIMD_PARAM_TAP_COUNT, UNI_SIMD_PARAM_BIN_COUNT,
                          UNI_SIMD_PARAM_DECIMATION, UNI_SIMD_PARAM_GRID_OFFSET,
                          UNI_SIMD_PARAM_LOGICAL_BINS, UNI_SIMD_PARAM_LOGICAL_BIN_COUNT}) {
        if (params.present[static_cast<std::size_t>(id)]) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] uni_simd_result_e to_c(const uni::simd::Result result) noexcept {
    switch (result) {
    case uni::simd::Result::success:
        return UNI_SIMD_RESULT_SUCCESS;
    case uni::simd::Result::invalid_argument:
        return UNI_SIMD_RESULT_INVALID_ARGUMENT;
    case uni::simd::Result::invalid_size:
        return UNI_SIMD_RESULT_INVALID_SIZE;
    case uni::simd::Result::overlapping_buffers:
        return UNI_SIMD_RESULT_OVERLAPPING_BUFFERS;
    case uni::simd::Result::unsupported_backend:
        return UNI_SIMD_RESULT_UNSUPPORTED_BACKEND;
    case uni::simd::Result::out_of_memory:
        return UNI_SIMD_RESULT_OUT_OF_MEMORY;
    }
    return UNI_SIMD_RESULT_INVALID_ARGUMENT;
}

[[nodiscard]] const uni::simd::Context* select_context(const ParsedParams& params,
                                                       uni_simd_result_e& error) noexcept {
    auto& cache = context_cache();
    if (params.math_mode == UNI_SIMD_MATH_DETERMINISTIC) {
        if (params.backend != UNI_SIMD_BACKEND_AUTOMATIC && params.backend != UNI_SIMD_BACKEND_GENERIC) {
            error = UNI_SIMD_RESULT_INVALID_ARGUMENT;
            return nullptr;
        }
        if (!cache.deterministic) {
            error = UNI_SIMD_RESULT_UNSUPPORTED_BACKEND;
            return nullptr;
        }
        return &*cache.deterministic;
    }
    if (params.prefer_energy_efficiency && params.backend == UNI_SIMD_BACKEND_AUTOMATIC) {
        if (!cache.energy_efficient) {
            error = UNI_SIMD_RESULT_UNSUPPORTED_BACKEND;
            return nullptr;
        }
        return &*cache.energy_efficient;
    }
    const auto index = static_cast<std::size_t>(params.backend);
    if (index >= cache.fast.size() || !cache.fast[index]) {
        error = index < cache.errors.size() ? to_c(cache.errors[index]) : UNI_SIMD_RESULT_INVALID_ARGUMENT;
        return nullptr;
    }
    return &*cache.fast[index];
}

[[nodiscard]] bool valid_read_buffer(const uni_simd_const_buffer_t& buffer) noexcept {
    return buffer.count == 0U || buffer.data != nullptr;
}

[[nodiscard]] bool valid_write_buffer(const uni_simd_buffer_t& buffer) noexcept {
    return buffer.count == 0U || buffer.data != nullptr;
}

[[nodiscard]] bool overlaps(const void* const left, const std::size_t left_bytes,
                            const void* const right, const std::size_t right_bytes) noexcept {
    if (left_bytes == 0U || right_bytes == 0U) {
        return false;
    }
    const auto left_begin = reinterpret_cast<std::uintptr_t>(left);
    const auto right_begin = reinterpret_cast<std::uintptr_t>(right);
    return left_begin <= right_begin ? right_begin - left_begin < left_bytes
                                     : left_begin - right_begin < right_bytes;
}

void set_resolved_backend(ParsedParams& params, const uni::simd::Backend backend) noexcept {
    if (params.resolved_backend != nullptr) {
        params.resolved_backend->value.u32 = static_cast<std::uint32_t>(backend);
    }
}

[[nodiscard]] std::optional<uni::simd::Kernel> primitive_kernel(const uni_simd_kernel_e kernel) noexcept {
    switch (kernel) {
    case UNI_SIMD_KERNEL_INVERT_LSB_U8:
        return uni::simd::Kernel::invert_lsb;
    case UNI_SIMD_KERNEL_INVERT_U8:
        return uni::simd::Kernel::invert_bytes;
    case UNI_SIMD_KERNEL_PACK_BITS_LSB_U8:
        return uni::simd::Kernel::pack_bits_lsb;
    case UNI_SIMD_KERNEL_PACK_BITS_MSB_U8:
        return uni::simd::Kernel::pack_bits_msb;
    case UNI_SIMD_KERNEL_UNPACK_BITS_LSB_U8:
        return uni::simd::Kernel::unpack_bits_lsb;
    case UNI_SIMD_KERNEL_UNPACK_BITS_MSB_U8:
        return uni::simd::Kernel::unpack_bits_msb;
    case UNI_SIMD_KERNEL_QUANTIZE_CF32_U8:
        return uni::simd::Kernel::quantize_interleaved_cf32_u8;
    case UNI_SIMD_KERNEL_MAGNITUDE_SQUARED_CF32_F32:
        return uni::simd::Kernel::magnitude_squared_cf32;
    case UNI_SIMD_KERNEL_POWER_SPECTRAL_DENSITY_CF32_F32:
        return uni::simd::Kernel::power_spectral_density_cf32;
    case UNI_SIMD_KERNEL_DOT_CF32_F32:
        return uni::simd::Kernel::dot_cf32_f32;
    case UNI_SIMD_KERNEL_DOT_SYMMETRIC_CF32_F32:
        return uni::simd::Kernel::dot_symmetric_cf32_f32;
    default:
        return std::nullopt;
    }
}

[[nodiscard]] uni_simd_result_e execute_stateless(const uni_simd_kernel_e kernel,
                                                  const void* const input,
                                                  void* const output,
                                                  ParsedParams& params,
                                                  const uni::simd::Context& context) noexcept {
    if (kernel == UNI_SIMD_KERNEL_IFFT_SPLIT_CF32) {
        if (input != nullptr || output == nullptr) {
            return UNI_SIMD_RESULT_INVALID_ARGUMENT;
        }
        auto& split = *static_cast<uni_simd_split_cf32_t*>(output);
        if (split.descriptor_size != sizeof(uni_simd_split_cf32_t)) {
            return UNI_SIMD_RESULT_INVALID_ARGUMENT;
        }
        if (split.transform_count != 0U && (split.real == nullptr || split.imag == nullptr)) {
            return UNI_SIMD_RESULT_INVALID_ARGUMENT;
        }
        const auto ifft = context.make_ifft_cf32(split.transform_size);
        if (!ifft) {
            return to_c(ifft.error());
        }
        const std::size_t stride = split.stride == 0U ? split.transform_size : split.stride;
        if (split.transform_count != 0U &&
            (stride < split.transform_size || split.transform_count - 1U >
                 (std::numeric_limits<std::size_t>::max() - split.transform_size) / stride)) {
            return UNI_SIMD_RESULT_INVALID_SIZE;
        }
        const std::size_t required = split.transform_count == 0U
                                         ? 0U
                                         : (split.transform_count - 1U) * stride + split.transform_size;
        const auto result = ifft->execute({.real = {split.real, required},
                                           .imag = {split.imag, required},
                                           .transform_count = split.transform_count,
                                           .stride = split.stride});
        if (result == uni::simd::Result::success) {
            set_resolved_backend(params, ifft->backend());
        }
        return to_c(result);
    }

    if (input == nullptr || output == nullptr) {
        return UNI_SIMD_RESULT_INVALID_ARGUMENT;
    }
    const auto& src = *static_cast<const uni_simd_const_buffer_t*>(input);
    auto& dst = *static_cast<uni_simd_buffer_t*>(output);
    if (!valid_read_buffer(src) || !valid_write_buffer(dst)) {
        return UNI_SIMD_RESULT_INVALID_ARGUMENT;
    }

    uni::simd::Result result = uni::simd::Result::invalid_argument;
    const auto source_bytes = std::span<const std::uint8_t>{static_cast<const std::uint8_t*>(src.data), src.count};
    auto destination_bytes = std::span<std::uint8_t>{static_cast<std::uint8_t*>(dst.data), dst.count};
    switch (kernel) {
    case UNI_SIMD_KERNEL_COPY_U8:
        result = context.copy(destination_bytes, source_bytes);
        break;
    case UNI_SIMD_KERNEL_INVERT_LSB_U8:
        result = context.invert_lsb(destination_bytes, source_bytes);
        break;
    case UNI_SIMD_KERNEL_INVERT_U8:
        result = context.invert_bytes(destination_bytes, source_bytes);
        break;
    case UNI_SIMD_KERNEL_PACK_BITS_LSB_U8:
        result = context.pack_bits_lsb(destination_bytes, source_bytes);
        break;
    case UNI_SIMD_KERNEL_PACK_BITS_MSB_U8:
        result = context.pack_bits_msb(destination_bytes, source_bytes);
        break;
    case UNI_SIMD_KERNEL_UNPACK_BITS_LSB_U8:
        result = context.unpack_bits_lsb(destination_bytes, source_bytes);
        break;
    case UNI_SIMD_KERNEL_UNPACK_BITS_MSB_U8:
        result = context.unpack_bits_msb(destination_bytes, source_bytes);
        break;
    case UNI_SIMD_KERNEL_QUANTIZE_CF32_U8:
        result = context.quantize_interleaved_cf32_u8_raw(
            destination_bytes, src.data, src.count, {.scale = params.scale, .offset = params.offset});
        break;
    case UNI_SIMD_KERNEL_MAGNITUDE_SQUARED_CF32_F32:
        result = context.magnitude_squared_raw(
            {static_cast<float*>(dst.data), dst.count}, src.data, src.count, params.normalization_factor);
        break;
    case UNI_SIMD_KERNEL_POWER_SPECTRAL_DENSITY_CF32_F32:
        if (!params.has_rbw_hz) {
            return UNI_SIMD_RESULT_INVALID_ARGUMENT;
        }
        result = context.power_spectral_density_raw(
            {static_cast<float*>(dst.data), dst.count}, src.data, src.count,
            params.normalization_factor, params.rbw_hz);
        break;
    case UNI_SIMD_KERNEL_DOT_CF32_F32: {
        if (!params.has_taps || !params.has_tap_count || (params.tap_count != 0U && params.taps == nullptr) ||
            dst.count < 1U || dst.data == nullptr) {
            return UNI_SIMD_RESULT_INVALID_ARGUMENT;
        }
        std::complex<float> value{};
        result = context.dot_cf32_f32_raw(value, src.data, src.count, {params.taps, params.tap_count});
        if (result == uni::simd::Result::success) {
            auto* const components = static_cast<float*>(dst.data);
            components[0] = value.real();
            components[1] = value.imag();
        }
        break;
    }
    case UNI_SIMD_KERNEL_DOT_SYMMETRIC_CF32_F32: {
        if (!params.has_taps || !params.has_tap_count || !params.has_center_tap ||
            (params.tap_count != 0U && params.taps == nullptr) || dst.count < 1U || dst.data == nullptr) {
            return UNI_SIMD_RESULT_INVALID_ARGUMENT;
        }
        std::complex<float> value{};
        result = context.dot_symmetric_cf32_f32_raw(
            value, src.data, src.count, {params.taps, params.tap_count}, params.center_tap);
        if (result == uni::simd::Result::success) {
            auto* const components = static_cast<float*>(dst.data);
            components[0] = value.real();
            components[1] = value.imag();
        }
        break;
    }
    default:
        return UNI_SIMD_RESULT_INVALID_ARGUMENT;
    }

    if (result == uni::simd::Result::success) {
        if (const auto operation = primitive_kernel(kernel)) {
            set_resolved_backend(params, context.kernel_backend(*operation));
        } else {
            set_resolved_backend(params, uni::simd::Backend::generic);
        }
    }
    return to_c(result);
}

[[nodiscard]] uni_simd_result_e execute_pfb(const void* const input, void* const output,
                                            ParsedParams& params,
                                            uni_simd_state_t** const state,
                                            const uni::simd::Context& context) {
    if (state == nullptr) {
        return UNI_SIMD_RESULT_INVALID_ARGUMENT;
    }
    std::unique_ptr<uni_simd_state_t> pending_state;
    uni_simd_state_t* active_state = *state;
    if (active_state == nullptr) {
        if (!params.has_bin_count || !params.has_decimation || !params.has_grid_offset || !params.has_taps ||
            !params.has_tap_count || !params.has_logical_bins || !params.has_logical_bin_count ||
            (params.tap_count != 0U && params.taps == nullptr) ||
            (params.logical_bin_count != 0U && params.logical_bins == nullptr)) {
            return UNI_SIMD_RESULT_INVALID_ARGUMENT;
        }
        auto pfb = context.make_pfb_channelizer({
            .bin_count = params.bin_count,
            .decimation = params.decimation,
            .grid_offset = static_cast<uni::simd::PfbGridOffset>(params.grid_offset),
            .taps = {params.taps, params.tap_count},
            .logical_bins = {params.logical_bins, params.logical_bin_count},
        });
        if (!pfb) {
            return to_c(pfb.error());
        }
        pending_state.reset(new (std::nothrow) uni_simd_state_t{.pfb = std::move(*pfb)});
        if (!pending_state) {
            return UNI_SIMD_RESULT_OUT_OF_MEMORY;
        }
        active_state = pending_state.get();
    }

    const auto commit_pending_state = [&] {
        if (pending_state) {
            *state = pending_state.release();
            runtime().state_count.fetch_add(1U, std::memory_order_relaxed);
        }
    };
    auto& active = *active_state;
    const auto complete_success = [&] {
        set_resolved_backend(params, active.pfb.backend());
        commit_pending_state();
        return UNI_SIMD_RESULT_SUCCESS;
    };
    if (params.reset && params.query_output_count) {
        return UNI_SIMD_RESULT_INVALID_ARGUMENT;
    }
    if (input == nullptr) {
        if (output != nullptr || params.query_output_count || (!pending_state && !params.reset)) {
            return UNI_SIMD_RESULT_INVALID_ARGUMENT;
        }
        if (params.reset) {
            const auto result = active.pfb.reset();
            if (result != uni::simd::Result::success) {
                return to_c(result);
            }
        }
        if (params.output_count != nullptr) {
            params.output_count->value.size = 0U;
        }
        return complete_success();
    }

    const auto& src = *static_cast<const uni_simd_const_buffer_t*>(input);
    if (!params.query_output_count && !valid_read_buffer(src)) {
        return UNI_SIMD_RESULT_INVALID_ARGUMENT;
    }
    const auto expected = params.reset
                              ? std::expected<std::size_t, uni::simd::Result>{
                                    src.count == 0U ? 0U : 1U + (src.count - 1U) / active.pfb.decimation()}
                              : active.pfb.output_count(src.count);
    if (!expected) {
        return to_c(expected.error());
    }
    if (params.query_output_count) {
        if (params.output_count == nullptr) {
            return UNI_SIMD_RESULT_INVALID_ARGUMENT;
        }
        params.output_count->value.size = *expected;
        return complete_success();
    }
    if (src.count > std::numeric_limits<std::size_t>::max() / (2U * sizeof(float))) {
        return UNI_SIMD_RESULT_INVALID_SIZE;
    }
    if (output == nullptr) {
        if (!active.pfb.logical_bins().empty()) {
            return UNI_SIMD_RESULT_INVALID_ARGUMENT;
        }
        if (params.reset) {
            const auto result = active.pfb.reset();
            if (result != uni::simd::Result::success) {
                return to_c(result);
            }
        }
        uni::simd::PfbChannelizerBlock block{
            .input = {static_cast<const float*>(src.data), src.count * 2U},
        };
        const auto produced = active.pfb.process(block);
        if (!produced) {
            return to_c(produced.error());
        }
        if (params.output_count != nullptr) {
            params.output_count->value.size = *produced;
        }
        return complete_success();
    }

    auto& destinations = *static_cast<uni_simd_buffer_array_t*>(output);
    if (destinations.count != active.pfb.logical_bins().size() ||
        (destinations.count != 0U && destinations.buffers == nullptr) ||
        *expected > std::numeric_limits<std::size_t>::max() / (2U * sizeof(float))) {
        return UNI_SIMD_RESULT_INVALID_ARGUMENT;
    }
    for (std::size_t index = 0U; index < destinations.count; ++index) {
        const auto& destination = destinations.buffers[index];
        if (!valid_write_buffer(destination) || destination.count < *expected) {
            return UNI_SIMD_RESULT_INVALID_SIZE;
        }
        if (overlaps(src.data, src.count * 2U * sizeof(float),
                     destination.data, *expected * 2U * sizeof(float))) {
            return UNI_SIMD_RESULT_OVERLAPPING_BUFFERS;
        }
        for (std::size_t previous = 0U; previous < index; ++previous) {
            if (overlaps(destinations.buffers[previous].data, *expected * 2U * sizeof(float),
                         destination.data, *expected * 2U * sizeof(float))) {
                return UNI_SIMD_RESULT_OVERLAPPING_BUFFERS;
            }
        }
    }

    if (params.reset) {
        const auto result = active.pfb.reset();
        if (result != uni::simd::Result::success) {
            return to_c(result);
        }
    }
    uni::simd::PfbChannelizerBlock block{
        .input = {static_cast<const float*>(src.data), src.count * 2U},
    };
    for (std::size_t output_index = 0U; output_index < destinations.count; ++output_index) {
        block.outputs[output_index] = {
            static_cast<float*>(destinations.buffers[output_index].data), *expected * 2U};
    }
    const auto produced = active.pfb.process(block);
    if (!produced) {
        return to_c(produced.error());
    }
    if (params.output_count != nullptr) {
        params.output_count->value.size = *produced;
    }
    return complete_success();
}

} // namespace

uni_simd_result_e UNI_SIMD_CALL uni_simd_initialize(void) {
    try {
        auto& active = runtime();
        const std::unique_lock lock{active.mutex};
        active.initialized = true;
        return UNI_SIMD_RESULT_SUCCESS;
    } catch (...) {
        return UNI_SIMD_RESULT_INVALID_STATE;
    }
}

uni_simd_result_e UNI_SIMD_CALL uni_simd_finalize(void) {
    try {
        auto& active = runtime();
        const std::unique_lock lock{active.mutex};
        if (active.state_count.load(std::memory_order_relaxed) != 0U) {
            return UNI_SIMD_RESULT_INVALID_STATE;
        }
        active.initialized = false;
        return UNI_SIMD_RESULT_SUCCESS;
    } catch (...) {
        return UNI_SIMD_RESULT_INVALID_STATE;
    }
}

void UNI_SIMD_CALL uni_simd_state_free(uni_simd_state_t* const state) {
    if (state == nullptr) {
        return;
    }
    delete state;
    runtime().state_count.fetch_sub(1U, std::memory_order_relaxed);
}

uni_simd_result_e UNI_SIMD_CALL uni_simd_execute(const uni_simd_kernel_e kernel, const void* const input,
                                                 void* const output, uni_simd_param_t* const params,
                                                 const size_t params_len, uni_simd_state_t** const state) {
    try {
        auto& active = runtime();
        const std::shared_lock lock{active.mutex};
        if (!active.initialized) {
            return UNI_SIMD_RESULT_NOT_INITIALIZED;
        }
        if (kernel <= UNI_SIMD_KERNEL_UNKNOWN || kernel > UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32) {
            return UNI_SIMD_RESULT_INVALID_ARGUMENT;
        }
        if (kernel != UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32 && state != nullptr && *state != nullptr) {
            return UNI_SIMD_RESULT_INVALID_STATE;
        }

        ParsedParams parsed;
        if (const auto parsed_result = parse_params(params, params_len, parsed);
            parsed_result != UNI_SIMD_RESULT_SUCCESS) {
            return parsed_result;
        }
        if (!parameters_allowed(kernel, parsed) ||
            (kernel == UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32 && state != nullptr && *state != nullptr &&
             has_pfb_creation_params(parsed))) {
            return UNI_SIMD_RESULT_INVALID_ARGUMENT;
        }
        uni_simd_result_e context_error = UNI_SIMD_RESULT_SUCCESS;
        const auto* const context = select_context(parsed, context_error);
        if (context == nullptr) {
            return context_error;
        }

        if (kernel == UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32) {
            return execute_pfb(input, output, parsed, state, *context);
        }
        return execute_stateless(kernel, input, output, parsed, *context);
    } catch (const std::bad_alloc&) {
        return UNI_SIMD_RESULT_OUT_OF_MEMORY;
    } catch (const std::length_error&) {
        return UNI_SIMD_RESULT_INVALID_SIZE;
    } catch (...) {
        return UNI_SIMD_RESULT_INVALID_ARGUMENT;
    }
}
