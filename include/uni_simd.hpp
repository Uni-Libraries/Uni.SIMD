// SPDX-License-Identifier: MIT

#pragma once

/**
 * C++23 interface of Uni.SIMD: RAII kernel handles and the prepared FIR and channelizer kernels.
 *
 * Header-only on purpose: it only calls the stable C API of uni_simd.h, so no C++ type crosses the
 * shared library boundary and any C++23 compiler can use it against any build of the library.
 * Results are reported as uni_simd_result_e, never as exceptions; failed factories return the
 * result through std::expected.
 */

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <new>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "uni_simd.h"

namespace Uni::SIMD {

struct KernelOptions final {
    uni_simd_backend_e backend{UNI_SIMD_BACKEND_AUTOMATIC};
    uni_simd_math_mode_e math_mode{UNI_SIMD_MATH_FAST};
    bool prefer_energy_efficiency{};
};

class KernelInstance final {
  public:
    KernelInstance() noexcept;
    ~KernelInstance();

    KernelInstance(const KernelInstance&) = delete;
    KernelInstance& operator=(const KernelInstance&) = delete;
    KernelInstance(KernelInstance&&) noexcept;
    KernelInstance& operator=(KernelInstance&&) noexcept;

    [[nodiscard]] static std::expected<KernelInstance, uni_simd_result_e> Create(uni_simd_kernel_e kernel, KernelOptions options = {}) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] uni_simd_backend_e ResolvedBackend() const noexcept;

    [[nodiscard]] uni_simd_result_e ParamSet(uni_simd_param_id id, std::uint32_t value) noexcept;
    [[nodiscard]] uni_simd_result_e ParamSet(uni_simd_param_id id, std::size_t value) noexcept;
    [[nodiscard]] uni_simd_result_e ParamSet(uni_simd_param_id id, float value) noexcept;
    [[nodiscard]] uni_simd_result_e ParamSet(uni_simd_param_id id, const void* value) noexcept;
    [[nodiscard]] uni_simd_result_e ParamSet(uni_simd_param_id id, void* value) noexcept;
    [[nodiscard]] uni_simd_result_e ParamSetMany(std::span<uni_simd_param_t> parameters) noexcept;
    [[nodiscard]] uni_simd_result_e Execute(const void* input, void* output) noexcept;
    [[nodiscard]] uni_simd_result_e Reset() noexcept;
    /** Take over the streaming state of `source` (same kernel and configuration); see uni_simd_kernel_copy_state(). */
    [[nodiscard]] uni_simd_result_e CopyStateFrom(const KernelInstance& source) noexcept;
    /** Symbol synchronizer only: move past `block`'s input without producing symbols; see uni_simd_symbol_sync4_skip(). */
    [[nodiscard]] uni_simd_result_e SymbolSync4Skip(const uni_simd_symbol_sync4_block_t& block) noexcept;
    /** QPSK Costas only: coast the oscillators past `sample_count` samples; see uni_simd_qpsk_costas4_skip(). */
    [[nodiscard]] uni_simd_result_e QpskCostas4Skip(std::size_t sample_count) noexcept;
    /** QPSK Costas only: move lane k's lock point by quarter_turns[k] * pi / 2; see uni_simd_qpsk_costas4_rotate(). */
    [[nodiscard]] uni_simd_result_e QpskCostas4Rotate(const std::array<std::int32_t, UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT>& quarter_turns) noexcept;

  private:
    struct Impl;
    explicit KernelInstance(std::unique_ptr<Impl> impl) noexcept;
    [[nodiscard]] uni_simd_kernel_t* native() const noexcept;

    std::unique_ptr<Impl> m_impl;

    friend class PfbChannelizer;
};

class PreparedDot final {
  public:
    PreparedDot() noexcept;
    ~PreparedDot();

    PreparedDot(const PreparedDot&) = delete;
    PreparedDot& operator=(const PreparedDot&) = delete;
    PreparedDot(PreparedDot&&) noexcept;
    PreparedDot& operator=(PreparedDot&&) noexcept;

    [[nodiscard]] static std::expected<PreparedDot, uni_simd_result_e> Create(std::span<const float> taps, KernelOptions options = {}) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] uni_simd_backend_e ResolvedBackend() const noexcept;
    [[nodiscard]] uni_simd_result_e Execute(std::span<const std::complex<float>> input, std::complex<float>& output) noexcept;

  private:
    struct Impl;
    explicit PreparedDot(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> m_impl;
};

class PreparedSymmetricDot final {
  public:
    PreparedSymmetricDot() noexcept;
    ~PreparedSymmetricDot();

    PreparedSymmetricDot(const PreparedSymmetricDot&) = delete;
    PreparedSymmetricDot& operator=(const PreparedSymmetricDot&) = delete;
    PreparedSymmetricDot(PreparedSymmetricDot&&) noexcept;
    PreparedSymmetricDot& operator=(PreparedSymmetricDot&&) noexcept;

    [[nodiscard]] static std::expected<PreparedSymmetricDot, uni_simd_result_e> Create(std::span<const float> tap_pairs, float center_tap,
                                                                                       KernelOptions options = {}) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] uni_simd_backend_e ResolvedBackend() const noexcept;
    [[nodiscard]] uni_simd_result_e Execute(std::span<const std::complex<float>> input, std::complex<float>& output) noexcept;

  private:
    struct Impl;
    explicit PreparedSymmetricDot(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> m_impl;
};

struct PfbConfiguration final {
    std::size_t bin_count{};
    std::size_t decimation{};
    uni_simd_pfb_grid_offset_e grid_offset{UNI_SIMD_PFB_INTEGER_BINS};
    std::span<const float> taps{};
    std::span<const std::int32_t> logical_bins{};
};

class PfbChannelizer final {
  public:
    PfbChannelizer() noexcept;
    ~PfbChannelizer();

    PfbChannelizer(const PfbChannelizer&) = delete;
    PfbChannelizer& operator=(const PfbChannelizer&) = delete;
    PfbChannelizer(PfbChannelizer&&) noexcept;
    PfbChannelizer& operator=(PfbChannelizer&&) noexcept;

    [[nodiscard]] static std::expected<PfbChannelizer, uni_simd_result_e> Create(const PfbConfiguration& config, KernelOptions options = {}) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] uni_simd_backend_e ResolvedBackend() const noexcept;
    [[nodiscard]] std::size_t OutputCount() const noexcept;

    [[nodiscard]] uni_simd_result_e Reset() noexcept;
    [[nodiscard]] uni_simd_result_e Predict(std::size_t input_count, std::size_t& output_count) noexcept;
    [[nodiscard]] uni_simd_result_e Process(std::span<const std::complex<float>> input, std::span<const std::span<std::complex<float>>> outputs,
                                            std::size_t& output_count, bool reset_before_process = false) noexcept;

    /** Continue the stream exactly where `other` (same configuration) currently is. */
    [[nodiscard]] uni_simd_result_e CopyStateFrom(const PfbChannelizer& other) noexcept;

    /**
     * Consume `input` without producing its outputs, as if it had been processed. Used to let a
     * second channelizer filter the tail of a block while this one filters the head.
     */
    [[nodiscard]] uni_simd_result_e Advance(std::span<const std::complex<float>> input) noexcept;

  private:
    struct Impl;
    explicit PfbChannelizer(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> m_impl;
};

/** Stable lower-case name of a backend, for logs and statistics. */
[[nodiscard]] std::string_view BackendName(uni_simd_backend_e backend) noexcept;

//
// Implementation
//

namespace Detail {

// The C API reports results and backends as fixed-width integers, its enumerators are plain enums.
inline constexpr uni_simd_result_e InvalidState = UNI_SIMD_RESULT_INVALID_STATE;
inline constexpr uni_simd_backend_e AutomaticBackend = UNI_SIMD_BACKEND_AUTOMATIC;


static_assert(sizeof(std::complex<float>) == UNI_SIMD_CF32_COMPONENT_COUNT * sizeof(float));

[[nodiscard]] inline uni_simd_result_e RuntimeInitialize() noexcept { return uni_simd_initialize(); }

[[nodiscard]] inline uni_simd_param_val U32Value(const std::uint32_t value) noexcept {
    uni_simd_param_val result{};
    result.u32 = value;
    return result;
}

[[nodiscard]] inline uni_simd_param_val SizeValue(const std::size_t value) noexcept {
    uni_simd_param_val result{};
    result.size = value;
    return result;
}

[[nodiscard]] inline uni_simd_param_val FloatValue(const float value) noexcept {
    uni_simd_param_val result{};
    result.f32 = value;
    return result;
}

[[nodiscard]] inline uni_simd_param_val ConstPointerValue(const void* const value) noexcept {
    uni_simd_param_val result{};
    result.const_pointer = value;
    return result;
}

[[nodiscard]] inline uni_simd_param_val PointerValue(void* const value) noexcept {
    uni_simd_param_val result{};
    result.pointer = value;
    return result;
}

[[nodiscard]] inline uni_simd_param_t Parameter(const uni_simd_param_id id, const uni_simd_param_val value) noexcept { return {.id = id, .value = value}; }

[[nodiscard]] inline bool Success(const uni_simd_result_e result) noexcept { return result == UNI_SIMD_RESULT_SUCCESS; }

} // namespace Detail

struct KernelInstance::Impl final {
    uni_simd_kernel_t* kernel{};
    uni_simd_backend_e resolved_backend{UNI_SIMD_BACKEND_AUTOMATIC};

    ~Impl() { (void)uni_simd_kernel_free(kernel); }
};

inline KernelInstance::KernelInstance() noexcept = default;
inline KernelInstance::KernelInstance(std::unique_ptr<Impl> impl) noexcept : m_impl(std::move(impl)) {}
inline KernelInstance::~KernelInstance() = default;
inline KernelInstance::KernelInstance(KernelInstance&&) noexcept = default;
inline KernelInstance& KernelInstance::operator=(KernelInstance&&) noexcept = default;

inline std::expected<KernelInstance, uni_simd_result_e> KernelInstance::Create(const uni_simd_kernel_e kernel, const KernelOptions options) noexcept {
    const auto initialized = Detail::RuntimeInitialize();
    if (!Detail::Success(initialized)) {
        return std::unexpected(initialized);
    }

    auto impl = std::unique_ptr<Impl>{new (std::nothrow) Impl{}};
    if (!impl) {
        return std::unexpected(UNI_SIMD_RESULT_OUT_OF_MEMORY);
    }
    impl->kernel = uni_simd_kernel_create(kernel);
    if (impl->kernel == nullptr) {
        return std::unexpected(kernel == UNI_SIMD_KERNEL_UNKNOWN ? UNI_SIMD_RESULT_INVALID_ARGUMENT : UNI_SIMD_RESULT_OUT_OF_MEMORY);
    }

    KernelInstance result{std::move(impl)};
    std::array parameters{
        Detail::Parameter(UNI_SIMD_PARAM_BACKEND, Detail::U32Value(options.backend)),
        Detail::Parameter(UNI_SIMD_PARAM_MATH_MODE, Detail::U32Value(options.math_mode)),
        Detail::Parameter(UNI_SIMD_PARAM_PREFER_ENERGY_EFFICIENCY, Detail::U32Value(static_cast<std::uint32_t>(options.prefer_energy_efficiency))),
        Detail::Parameter(UNI_SIMD_PARAM_RESOLVED_BACKEND, Detail::PointerValue(&result.m_impl->resolved_backend)),
    };
    if (const auto set = result.ParamSetMany(parameters); !Detail::Success(set)) {
        return std::unexpected(set);
    }
    return result;
}

inline KernelInstance::operator bool() const noexcept { return m_impl && m_impl->kernel != nullptr; }

inline uni_simd_backend_e KernelInstance::ResolvedBackend() const noexcept { return m_impl ? m_impl->resolved_backend : Detail::AutomaticBackend; }

inline uni_simd_result_e KernelInstance::ParamSet(const uni_simd_param_id id, const std::uint32_t value) noexcept {
    return m_impl ? uni_simd_kernel_param_set(m_impl->kernel, Detail::Parameter(id, Detail::U32Value(value))) : Detail::InvalidState;
}

inline uni_simd_result_e KernelInstance::ParamSet(const uni_simd_param_id id, const std::size_t value) noexcept {
    return m_impl ? uni_simd_kernel_param_set(m_impl->kernel, Detail::Parameter(id, Detail::SizeValue(value))) : Detail::InvalidState;
}

inline uni_simd_result_e KernelInstance::ParamSet(const uni_simd_param_id id, const float value) noexcept {
    return m_impl ? uni_simd_kernel_param_set(m_impl->kernel, Detail::Parameter(id, Detail::FloatValue(value))) : Detail::InvalidState;
}

inline uni_simd_result_e KernelInstance::ParamSet(const uni_simd_param_id id, const void* const value) noexcept {
    return m_impl ? uni_simd_kernel_param_set(m_impl->kernel, Detail::Parameter(id, Detail::ConstPointerValue(value))) : Detail::InvalidState;
}

inline uni_simd_result_e KernelInstance::ParamSet(const uni_simd_param_id id, void* const value) noexcept {
    return m_impl ? uni_simd_kernel_param_set(m_impl->kernel, Detail::Parameter(id, Detail::PointerValue(value))) : Detail::InvalidState;
}

inline uni_simd_result_e KernelInstance::ParamSetMany(const std::span<uni_simd_param_t> parameters) noexcept {
    return m_impl ? uni_simd_kernel_param_set_many(m_impl->kernel, parameters.data(), parameters.size()) : Detail::InvalidState;
}

inline uni_simd_result_e KernelInstance::Execute(const void* const input, void* const output) noexcept {
    return m_impl ? uni_simd_kernel_execute(m_impl->kernel, input, output) : Detail::InvalidState;
}

inline uni_simd_result_e KernelInstance::Reset() noexcept { return m_impl ? uni_simd_kernel_reset(m_impl->kernel) : Detail::InvalidState; }

struct PreparedDot::Impl final {
    std::vector<float> taps;
    KernelInstance kernel;
};

inline PreparedDot::PreparedDot() noexcept = default;
inline PreparedDot::PreparedDot(std::unique_ptr<Impl> impl) noexcept : m_impl(std::move(impl)) {}
inline PreparedDot::~PreparedDot() = default;
inline PreparedDot::PreparedDot(PreparedDot&&) noexcept = default;
inline PreparedDot& PreparedDot::operator=(PreparedDot&&) noexcept = default;

inline std::expected<PreparedDot, uni_simd_result_e> PreparedDot::Create(const std::span<const float> taps, const KernelOptions options) noexcept {
    try {
        auto kernel = KernelInstance::Create(UNI_SIMD_KERNEL_DOT_CF32_F32, options);
        if (!kernel) {
            return std::unexpected(kernel.error());
        }
        auto impl = std::make_unique<Impl>(Impl{.taps = {taps.begin(), taps.end()}, .kernel = std::move(*kernel)});
        std::array parameters{
            Detail::Parameter(UNI_SIMD_PARAM_TAPS, Detail::ConstPointerValue(impl->taps.data())),
            Detail::Parameter(UNI_SIMD_PARAM_TAP_COUNT, Detail::SizeValue(impl->taps.size())),
        };
        if (const auto set = impl->kernel.ParamSetMany(parameters); !Detail::Success(set)) {
            return std::unexpected(set);
        }
        return PreparedDot{std::move(impl)};
    } catch (const std::bad_alloc&) {
        return std::unexpected(UNI_SIMD_RESULT_OUT_OF_MEMORY);
    }
}

inline PreparedDot::operator bool() const noexcept { return m_impl && m_impl->kernel; }
inline uni_simd_backend_e PreparedDot::ResolvedBackend() const noexcept { return m_impl ? m_impl->kernel.ResolvedBackend() : Detail::AutomaticBackend; }

inline uni_simd_result_e PreparedDot::Execute(const std::span<const std::complex<float>> input, std::complex<float>& output) noexcept {
    if (!m_impl) {
        return UNI_SIMD_RESULT_INVALID_STATE;
    }
    const uni_simd_const_buffer_t input_buffer{input.data(), input.size()};
    std::array<float, UNI_SIMD_CF32_COMPONENT_COUNT> components{};
    uni_simd_buffer_t output_buffer{components.data(), 1U};
    const auto result = m_impl->kernel.Execute(&input_buffer, &output_buffer);
    if (Detail::Success(result)) {
        output = {components[0], components[1]};
    }
    return result;
}

struct PreparedSymmetricDot::Impl final {
    std::vector<float> taps;
    KernelInstance kernel;
};

inline PreparedSymmetricDot::PreparedSymmetricDot() noexcept = default;
inline PreparedSymmetricDot::PreparedSymmetricDot(std::unique_ptr<Impl> impl) noexcept : m_impl(std::move(impl)) {}
inline PreparedSymmetricDot::~PreparedSymmetricDot() = default;
inline PreparedSymmetricDot::PreparedSymmetricDot(PreparedSymmetricDot&&) noexcept = default;
inline PreparedSymmetricDot& PreparedSymmetricDot::operator=(PreparedSymmetricDot&&) noexcept = default;

inline std::expected<PreparedSymmetricDot, uni_simd_result_e> PreparedSymmetricDot::Create(const std::span<const float> tap_pairs, const float center_tap,
                                                                                           const KernelOptions options) noexcept {
    try {
        auto kernel = KernelInstance::Create(UNI_SIMD_KERNEL_DOT_SYMMETRIC_CF32_F32, options);
        if (!kernel) {
            return std::unexpected(kernel.error());
        }
        auto impl = std::make_unique<Impl>(Impl{.taps = {tap_pairs.begin(), tap_pairs.end()}, .kernel = std::move(*kernel)});
        std::array parameters{
            Detail::Parameter(UNI_SIMD_PARAM_TAPS, Detail::ConstPointerValue(impl->taps.data())),
            Detail::Parameter(UNI_SIMD_PARAM_TAP_COUNT, Detail::SizeValue(impl->taps.size())),
            Detail::Parameter(UNI_SIMD_PARAM_CENTER_TAP, Detail::FloatValue(center_tap)),
        };
        if (const auto set = impl->kernel.ParamSetMany(parameters); !Detail::Success(set)) {
            return std::unexpected(set);
        }
        return PreparedSymmetricDot{std::move(impl)};
    } catch (const std::bad_alloc&) {
        return std::unexpected(UNI_SIMD_RESULT_OUT_OF_MEMORY);
    }
}

inline PreparedSymmetricDot::operator bool() const noexcept { return m_impl && m_impl->kernel; }
inline uni_simd_backend_e PreparedSymmetricDot::ResolvedBackend() const noexcept {
    return m_impl ? m_impl->kernel.ResolvedBackend() : Detail::AutomaticBackend;
}

inline uni_simd_result_e PreparedSymmetricDot::Execute(const std::span<const std::complex<float>> input, std::complex<float>& output) noexcept {
    if (!m_impl) {
        return UNI_SIMD_RESULT_INVALID_STATE;
    }
    const uni_simd_const_buffer_t input_buffer{input.data(), input.size()};
    std::array<float, UNI_SIMD_CF32_COMPONENT_COUNT> components{};
    uni_simd_buffer_t output_buffer{components.data(), 1U};
    const auto result = m_impl->kernel.Execute(&input_buffer, &output_buffer);
    if (Detail::Success(result)) {
        output = {components[0], components[1]};
    }
    return result;
}

struct PfbChannelizer::Impl final {
    std::vector<float> taps;
    std::vector<std::int32_t> logical_bins;
    std::vector<uni_simd_buffer_t> output_buffers;
    std::size_t produced{};
    KernelInstance kernel;
};

inline PfbChannelizer::PfbChannelizer() noexcept = default;
inline PfbChannelizer::PfbChannelizer(std::unique_ptr<Impl> impl) noexcept : m_impl(std::move(impl)) {}
inline PfbChannelizer::~PfbChannelizer() = default;
inline PfbChannelizer::PfbChannelizer(PfbChannelizer&&) noexcept = default;
inline PfbChannelizer& PfbChannelizer::operator=(PfbChannelizer&&) noexcept = default;

inline std::expected<PfbChannelizer, uni_simd_result_e> PfbChannelizer::Create(const PfbConfiguration& config, const KernelOptions options) noexcept {
    try {
        auto kernel = KernelInstance::Create(UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32, options);
        if (!kernel) {
            return std::unexpected(kernel.error());
        }
        auto impl = std::make_unique<Impl>(Impl{
            .taps = {config.taps.begin(), config.taps.end()},
            .logical_bins = {config.logical_bins.begin(), config.logical_bins.end()},
            .output_buffers = std::vector<uni_simd_buffer_t>(config.logical_bins.size()),
            .kernel = std::move(*kernel),
        });

        std::array parameters{
            Detail::Parameter(UNI_SIMD_PARAM_BIN_COUNT, Detail::SizeValue(config.bin_count)),
            Detail::Parameter(UNI_SIMD_PARAM_DECIMATION, Detail::SizeValue(config.decimation)),
            Detail::Parameter(UNI_SIMD_PARAM_GRID_OFFSET, Detail::U32Value(config.grid_offset)),
            Detail::Parameter(UNI_SIMD_PARAM_TAPS, Detail::ConstPointerValue(impl->taps.data())),
            Detail::Parameter(UNI_SIMD_PARAM_TAP_COUNT, Detail::SizeValue(impl->taps.size())),
            Detail::Parameter(UNI_SIMD_PARAM_LOGICAL_BINS, Detail::ConstPointerValue(impl->logical_bins.data())),
            Detail::Parameter(UNI_SIMD_PARAM_LOGICAL_BIN_COUNT, Detail::SizeValue(impl->logical_bins.size())),
            Detail::Parameter(UNI_SIMD_PARAM_OUTPUT_COUNT, Detail::PointerValue(&impl->produced)),
        };
        if (const auto configured = impl->kernel.ParamSetMany(parameters); !Detail::Success(configured)) {
            return std::unexpected(configured);
        }

        if (const auto prewarm = impl->kernel.Execute(nullptr, nullptr); !Detail::Success(prewarm)) {
            return std::unexpected(prewarm);
        }
        return PfbChannelizer{std::move(impl)};
    } catch (const std::bad_alloc&) {
        return std::unexpected(UNI_SIMD_RESULT_OUT_OF_MEMORY);
    }
}

inline PfbChannelizer::operator bool() const noexcept { return m_impl && m_impl->kernel; }
inline uni_simd_backend_e PfbChannelizer::ResolvedBackend() const noexcept { return m_impl ? m_impl->kernel.ResolvedBackend() : Detail::AutomaticBackend; }
inline std::size_t PfbChannelizer::OutputCount() const noexcept { return m_impl ? m_impl->logical_bins.size() : 0U; }

inline uni_simd_kernel_t* KernelInstance::native() const noexcept { return m_impl ? m_impl->kernel : nullptr; }

inline uni_simd_result_e KernelInstance::CopyStateFrom(const KernelInstance& source) noexcept {
    return native() && source.native() ? uni_simd_kernel_copy_state(native(), source.native()) : Detail::InvalidState;
}

inline uni_simd_result_e KernelInstance::SymbolSync4Skip(const uni_simd_symbol_sync4_block_t& block) noexcept {
    return native() ? uni_simd_symbol_sync4_skip(native(), &block) : Detail::InvalidState;
}

inline uni_simd_result_e KernelInstance::QpskCostas4Skip(const std::size_t sample_count) noexcept {
    return native() ? uni_simd_qpsk_costas4_skip(native(), sample_count) : Detail::InvalidState;
}

inline uni_simd_result_e KernelInstance::QpskCostas4Rotate(const std::array<std::int32_t, UNI_SIMD_QPSK_COSTAS4_CHANNEL_COUNT>& quarter_turns) noexcept {
    return native() ? uni_simd_qpsk_costas4_rotate(native(), quarter_turns.data()) : Detail::InvalidState;
}

inline uni_simd_result_e PfbChannelizer::CopyStateFrom(const PfbChannelizer& other) noexcept {
    if (!m_impl || !other.m_impl) {
        return UNI_SIMD_RESULT_INVALID_STATE;
    }
    return uni_simd_pfb_channelizer_copy_state(m_impl->kernel.native(), other.m_impl->kernel.native());
}

inline uni_simd_result_e PfbChannelizer::Advance(const std::span<const std::complex<float>> input) noexcept {
    if (!m_impl) {
        return UNI_SIMD_RESULT_INVALID_STATE;
    }
    const uni_simd_const_buffer_t buffer{input.data(), input.size()};
    return uni_simd_pfb_channelizer_advance(m_impl->kernel.native(), &buffer);
}

inline uni_simd_result_e PfbChannelizer::Reset() noexcept { return m_impl ? m_impl->kernel.Reset() : Detail::InvalidState; }

inline uni_simd_result_e PfbChannelizer::Predict(const std::size_t input_count, std::size_t& output_count) noexcept {
    if (!m_impl) {
        return UNI_SIMD_RESULT_INVALID_STATE;
    }
    const auto set = m_impl->kernel.ParamSet(UNI_SIMD_PARAM_QUERY_OUTPUT_COUNT, std::uint32_t{1U});
    if (!Detail::Success(set)) {
        return set;
    }
    const uni_simd_const_buffer_t input{nullptr, input_count};
    const auto result = m_impl->kernel.Execute(&input, nullptr);
    if (Detail::Success(result)) {
        output_count = m_impl->produced;
    }
    return result;
}

inline uni_simd_result_e PfbChannelizer::Process(const std::span<const std::complex<float>> input,
                                                 const std::span<const std::span<std::complex<float>>> outputs, std::size_t& output_count,
                                                 const bool reset_before_process) noexcept {
    if (!m_impl) {
        return UNI_SIMD_RESULT_INVALID_STATE;
    }
    if (outputs.size() != m_impl->output_buffers.size()) {
        return UNI_SIMD_RESULT_INVALID_SIZE;
    }
    for (std::size_t index = 0U; index < outputs.size(); ++index) {
        m_impl->output_buffers[index] = {.data = outputs[index].data(), .count = outputs[index].size()};
    }

    if (reset_before_process) {
        const auto set = m_impl->kernel.ParamSet(UNI_SIMD_PARAM_RESET, std::uint32_t{1U});
        if (!Detail::Success(set)) {
            return set;
        }
    }
    const uni_simd_const_buffer_t input_buffer{input.data(), input.size()};
    uni_simd_buffer_array_t output_array{m_impl->output_buffers.data(), m_impl->output_buffers.size()};
    const auto result = m_impl->kernel.Execute(&input_buffer, outputs.empty() ? nullptr : &output_array);
    if (Detail::Success(result)) {
        output_count = m_impl->produced;
    }
    return result;
}

inline std::string_view BackendName(const uni_simd_backend_e backend) noexcept {
    switch (backend) {
    case UNI_SIMD_BACKEND_AUTOMATIC:
        return "automatic";
    case UNI_SIMD_BACKEND_GENERIC:
        return "generic";
    case UNI_SIMD_BACKEND_X86_SSE2:
        return "sse2";
    case UNI_SIMD_BACKEND_X86_AVX2:
        return "avx2";
    case UNI_SIMD_BACKEND_X86_AVX2_FMA:
        return "avx2_fma";
    case UNI_SIMD_BACKEND_X86_AVX512:
        return "avx512";
    case UNI_SIMD_BACKEND_AARCH64_NEON:
        return "neon";
    default:
        return "unknown";
    }
}

} // namespace Uni::SIMD
