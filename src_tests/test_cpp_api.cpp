#ifdef NDEBUG
#undef NDEBUG
#endif

#include <array>
#include <cassert>
#include <complex>
#include <cstdint>
#include <span>
#include <utility>

#include "uni_simd.hpp"

namespace {

[[nodiscard]] Uni::SIMD::KernelOptions GenericOptions() noexcept { return {.backend = UNI_SIMD_BACKEND_GENERIC}; }

[[nodiscard]] uni_simd_result_e ExecuteBuffer(Uni::SIMD::KernelInstance& kernel, const void* source, const std::size_t source_count, void* destination,
                                              const std::size_t destination_count) {
    const uni_simd_const_buffer_t input{source, source_count};
    uni_simd_buffer_t output{destination, destination_count};
    return kernel.Execute(&input, &output);
}

} // namespace

// SIMD Pack8 enforces source and destination byte lengths
static void Test0() {
    constexpr std::array<std::uint8_t, 16U> source{
        1U, 0U, 1U, 1U, 0U, 0U, 1U, 0U, 0U, 1U, 1U, 0U, 1U, 0U, 0U, 1U,
    };
    std::array<std::uint8_t, 2U> output{};
    auto lsb = Uni::SIMD::KernelInstance::Create(UNI_SIMD_KERNEL_PACK_BITS_LSB_U8, GenericOptions());
    auto msb = Uni::SIMD::KernelInstance::Create(UNI_SIMD_KERNEL_PACK_BITS_MSB_U8, GenericOptions());
    assert(lsb.has_value());
    assert(msb.has_value());

    assert(ExecuteBuffer(*lsb, source.data(), source.size() - 1U, output.data(), output.size()) == UNI_SIMD_RESULT_INVALID_SIZE);
    assert(ExecuteBuffer(*lsb, source.data(), source.size(), output.data(), output.size() - 1U) == UNI_SIMD_RESULT_INVALID_SIZE);
    assert(ExecuteBuffer(*lsb, source.data(), source.size(), output.data(), output.size()) == UNI_SIMD_RESULT_SUCCESS);
    assert(output == std::array<std::uint8_t, 2U>{0x4dU, 0x96U});

    assert(ExecuteBuffer(*msb, source.data(), source.size(), output.data(), output.size()) == UNI_SIMD_RESULT_SUCCESS);
    assert(output == std::array<std::uint8_t, 2U>{0xb2U, 0x69U});
}

// SIMD prepared expansion and quantization kernels validate buffer sizes
static void Test1() {
    constexpr std::array<std::uint8_t, 2U> packed{0x4dU, 0x96U};
    std::array<std::uint8_t, 15U> unpacked{};
    auto unpack = Uni::SIMD::KernelInstance::Create(UNI_SIMD_KERNEL_UNPACK_BITS_LSB_U8, GenericOptions());
    assert(unpack.has_value());
    assert(ExecuteBuffer(*unpack, packed.data(), packed.size(), unpacked.data(), unpacked.size()) == UNI_SIMD_RESULT_INVALID_SIZE);

    constexpr std::array<std::complex<float>, 2U> symbols{{{1.0F, -1.0F}, {0.5F, -0.5F}}};
    std::array<std::uint8_t, 3U> soft{};
    auto quantize = Uni::SIMD::KernelInstance::Create(UNI_SIMD_KERNEL_QUANTIZE_CF32_U8, GenericOptions());
    assert(quantize.has_value());
    assert(quantize->ParamSet(UNI_SIMD_PARAM_SCALE, -10.0F) == UNI_SIMD_RESULT_SUCCESS);
    assert(ExecuteBuffer(*quantize, symbols.data(), symbols.size(), soft.data(), soft.size()) == UNI_SIMD_RESULT_INVALID_SIZE);

    const uni_simd_const_buffer_t empty_input{nullptr, 0U};
    uni_simd_buffer_t empty_output{nullptr, 0U};
    assert(unpack->Execute(&empty_input, &empty_output) == UNI_SIMD_RESULT_SUCCESS);
    assert(quantize->Execute(&empty_input, &empty_output) == UNI_SIMD_RESULT_SUCCESS);
}

// SIMD prepared kernels preserve fixed parameters across moves
static void Test2() {
    constexpr std::array taps{0.5F, 1.0F};
    constexpr std::array<std::complex<float>, 2U> input{{{1.0F, -1.0F}, {0.5F, -0.5F}}};
    auto created = Uni::SIMD::PreparedDot::Create(taps, GenericOptions());
    assert(created.has_value());
    auto dot = std::move(*created);
    std::complex<float> output{};
    assert(dot.Execute(input, output) == UNI_SIMD_RESULT_SUCCESS);
    assert(output == std::complex<float>{1.0F, -1.0F});
    assert(dot.ResolvedBackend() == UNI_SIMD_BACKEND_GENERIC);
}

// SIMD opaque PFB preserves counts and one-shot reset semantics
static void Test3() {
    constexpr std::array taps{1.0F};
    constexpr std::array<std::int32_t, 1U> logical_bins{0};
    auto created = Uni::SIMD::PfbChannelizer::Create(
        {.bin_count = 8U, .decimation = 4U, .grid_offset = UNI_SIMD_PFB_INTEGER_BINS, .taps = taps, .logical_bins = logical_bins}, GenericOptions());
    assert(created.has_value());
    auto channelizer = std::move(*created);
    assert(channelizer.ResolvedBackend() == UNI_SIMD_BACKEND_GENERIC);
    assert(channelizer.OutputCount() == 1U);

    std::size_t predicted = 0U;
    assert(channelizer.Predict(5U, predicted) == UNI_SIMD_RESULT_SUCCESS);
    assert(predicted == 2U);
    predicted = 0U;
    assert(channelizer.Predict(5U, predicted) == UNI_SIMD_RESULT_SUCCESS);
    assert(predicted == 2U);

    constexpr std::array<std::complex<float>, 5U> input{{
        {1.0F, -0.5F},
        {0.0F, 0.0F},
        {0.0F, 0.0F},
        {0.0F, 0.0F},
        {2.0F, 0.25F},
    }};
    std::array<std::complex<float>, 2U> output{};
    const std::array<std::span<std::complex<float>>, 1U> outputs{std::span{output}};
    std::size_t produced = 0U;
    assert(channelizer.Process(input, outputs, produced, true) == UNI_SIMD_RESULT_SUCCESS);
    assert(produced == output.size());
    assert(output[0] == input[0]);
    assert(output[1] == input[4]);

    constexpr std::array<std::complex<float>, 1U> failed_input{{{99.0F, 0.0F}}};
    std::array<std::complex<float>, 0U> no_capacity{};
    const std::array<std::span<std::complex<float>>, 1U> failed_outputs{std::span{no_capacity}};
    assert(channelizer.Process(failed_input, failed_outputs, produced, true) == UNI_SIMD_RESULT_INVALID_SIZE);

    constexpr std::array<std::complex<float>, 4U> continuation{{
        {10.0F, 0.0F},
        {20.0F, 0.0F},
        {30.0F, 0.0F},
        {40.0F, 0.0F},
    }};
    std::array<std::complex<float>, 1U> continuation_output{};
    const std::array<std::span<std::complex<float>>, 1U> continuation_outputs{std::span{continuation_output}};
    assert(channelizer.Process(continuation, continuation_outputs, produced) == UNI_SIMD_RESULT_SUCCESS);
    assert(continuation_output[0] == continuation.back());
    assert(channelizer.Reset() == UNI_SIMD_RESULT_SUCCESS);
}

int main() {
    Test0();
    Test1();
    Test2();
    Test3();
    return 0;
}
