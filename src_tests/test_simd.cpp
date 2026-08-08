#ifdef NDEBUG
#undef NDEBUG
#endif

#include <uni/simd/simd.hpp>

#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <span>

int main() {
    using uni::simd::Backend;
    using uni::simd::Result;

    const auto generic_result = uni::simd::create_context({.backend = Backend::generic});
    assert(generic_result.has_value());
    const auto& generic = *generic_result;

    constexpr std::array<std::uint8_t, 16U> bits{1U, 0U, 1U, 1U, 0U, 0U, 1U, 0U, 0U, 1U, 1U, 0U, 1U, 0U, 0U, 1U};
    std::array<std::uint8_t, 2U> packed{};
    assert(generic.pack_bits_lsb(packed, bits) == Result::success);
    assert((packed == std::array<std::uint8_t, 2U>{0x4dU, 0x96U}));
    assert(generic.pack_bits_msb(packed, bits) == Result::success);
    assert((packed == std::array<std::uint8_t, 2U>{0xb2U, 0x69U}));

    std::array<std::uint8_t, 15U> undersized{};
    assert(generic.unpack_bits_lsb(undersized, packed) == Result::invalid_size);
    assert(generic.unpack_bits_msb(undersized, packed) == Result::invalid_size);

    constexpr std::array<std::complex<float>, 2U> symbols{{{1.0f, -1.0f}, {0.5f, -0.5f}}};
    std::array<std::uint8_t, 3U> short_soft{};
    assert(generic.quantize_interleaved_cf32_u8(short_soft, symbols) == Result::invalid_size);
    std::array<std::uint8_t, 4U> soft{};
    assert(generic.quantize_interleaved_cf32_u8(soft, symbols, {.scale = -10.0f}) == Result::success);
    assert((soft == std::array<std::uint8_t, 4U>{118U, 138U, 123U, 133U}));

    std::array<float, 2U> power{};
    assert(generic.magnitude_squared(power, symbols, 1.0e-30f) == Result::success);
    assert(std::isinf(power[0]));
    const std::array<std::complex<float>, 1U> zero{{{0.0f, 0.0f}}};
    std::array<float, 1U> zero_power{};
    assert(generic.magnitude_squared(zero_power, zero, 1.0e-30f) == Result::success);
    assert(zero_power[0] == 0.0f);
    constexpr std::array<std::complex<float>, 1U> wide{{{1.0e20f, 0.0f}}};
    std::array<float, 1U> wide_power{};
    assert(generic.magnitude_squared(wide_power, wide, 1.0e20f) == Result::success);
    assert(std::isfinite(wide_power[0]));
    assert(std::abs(wide_power[0] - 1.0f) < 1.0e-6f);
    assert(generic.power_spectral_density(wide_power, wide, 1.0f, 1.0e20f) == Result::success);
    assert(std::isfinite(wide_power[0]));
    assert(std::abs(wide_power[0] / 1.0e20f - 1.0f) < 1.0e-6f);

    const auto automatic = uni::simd::create_context();
    assert(automatic.has_value());
    std::array<std::uint8_t, 16U> generic_unpacked{};
    std::array<std::uint8_t, 16U> automatic_unpacked{};
    assert(generic.unpack_bits_msb(generic_unpacked, packed) == Result::success);
    assert(automatic->unpack_bits_msb(automatic_unpacked, packed) == Result::success);
    assert(generic_unpacked == automatic_unpacked);

    std::array<std::uint8_t, 128U> vector_bits{};
    for (std::size_t i = 0; i < vector_bits.size(); ++i) {
        vector_bits[i] = static_cast<std::uint8_t>((i * 5U + 1U) & 1U);
    }
    std::array<std::uint8_t, 16U> vector_packed{};
    std::array<std::uint8_t, 128U> vector_unpacked{};
    assert(generic.pack_bits_lsb(vector_packed, vector_bits) == Result::success);
    assert(generic.unpack_bits_lsb(vector_unpacked, vector_packed) == Result::success);
    assert(vector_unpacked == vector_bits);

    std::array<std::complex<float>, 17U> vector_symbols{};
    std::array<float, 17U> vector_taps{};
    for (std::size_t i = 0; i < vector_symbols.size(); ++i) {
        vector_symbols[i] = {static_cast<float>(i) * 0.25f - 2.0f, static_cast<float>(i) * -0.125f + 1.0f};
        vector_taps[i] = static_cast<float>(i + 1U) / 19.0f;
    }
    std::array<std::uint8_t, 34U> vector_soft_reference{};
    std::array<float, 17U> vector_power_reference{};
    std::complex<float> vector_dot_reference{};
    std::complex<float> vector_symmetric_dot_reference{};
    assert(generic.quantize_interleaved_cf32_u8(vector_soft_reference, vector_symbols, {.scale = -7.0f}) == Result::success);
    assert(generic.magnitude_squared(vector_power_reference, vector_symbols, 3.0f) == Result::success);
    assert(generic.dot_cf32_f32(vector_dot_reference, vector_symbols, vector_taps) == Result::success);
    assert(generic.dot_symmetric_cf32_f32(vector_symmetric_dot_reference, vector_symbols,
                                          std::span<const float>{vector_taps}.first(8U), 0.25f) == Result::success);

    for (const auto backend : {Backend::generic, Backend::sse2, Backend::avx2, Backend::avx2_fma, Backend::avx512}) {
        const auto forced = uni::simd::create_context({.backend = backend});
        if (!forced.has_value()) {
            assert(forced.error() == Result::unsupported_backend);
            continue;
        }
        std::array<std::uint8_t, 16U> forced_unpacked{};
        std::array<std::uint8_t, 4U> forced_soft{};
        std::array<float, 2U> forced_power{};
        assert(forced->unpack_bits_msb(forced_unpacked, packed) == Result::success);
        assert(forced_unpacked == generic_unpacked);
        assert(forced->quantize_interleaved_cf32_u8(forced_soft, symbols, {.scale = -10.0f}) == Result::success);
        assert(forced_soft == soft);
        assert(forced->magnitude_squared(forced_power, symbols) == Result::success);
        assert(forced_power[0] == 2.0f);
        assert(forced_power[1] == 0.5f);

        std::array<std::uint8_t, 16U> backend_packed{};
        std::array<std::uint8_t, 128U> backend_unpacked{};
        std::array<std::uint8_t, 34U> backend_soft{};
        std::array<float, 17U> backend_power{};
        std::complex<float> backend_dot{};
        std::complex<float> backend_symmetric_dot{};
        assert(forced->pack_bits_lsb(backend_packed, vector_bits) == Result::success);
        assert(backend_packed == vector_packed);
        assert(forced->unpack_bits_lsb(backend_unpacked, backend_packed) == Result::success);
        assert(backend_unpacked == vector_bits);
        assert(forced->quantize_interleaved_cf32_u8(backend_soft, vector_symbols, {.scale = -7.0f}) == Result::success);
        assert(backend_soft == vector_soft_reference);
        assert(forced->magnitude_squared(backend_power, vector_symbols, 3.0f) == Result::success);
        for (std::size_t i = 0; i < backend_power.size(); ++i) {
            assert(std::abs(backend_power[i] - vector_power_reference[i]) < 1.0e-5f);
        }
        assert(forced->dot_cf32_f32(backend_dot, vector_symbols, vector_taps) == Result::success);
        assert(std::abs(backend_dot.real() - vector_dot_reference.real()) < 1.0e-4f);
        assert(std::abs(backend_dot.imag() - vector_dot_reference.imag()) < 1.0e-4f);
        assert(forced->dot_symmetric_cf32_f32(backend_symmetric_dot, vector_symbols,
                                              std::span<const float>{vector_taps}.first(8U), 0.25f) == Result::success);
        assert(std::abs(backend_symmetric_dot.real() - vector_symmetric_dot_reference.real()) < 1.0e-4f);
        assert(std::abs(backend_symmetric_dot.imag() - vector_symmetric_dot_reference.imag()) < 1.0e-4f);
    }

    constexpr std::array<std::complex<float>, 1U> exceptional{{{std::numeric_limits<float>::quiet_NaN(),
                                                                 std::numeric_limits<float>::infinity()}}};
    std::array<std::uint8_t, 2U> exceptional_soft{};
    assert(automatic->quantize_interleaved_cf32_u8(exceptional_soft, exceptional, {.scale = -1.0f}) == Result::success);
    assert(exceptional_soft[0] == 128U);
    assert(exceptional_soft[1] == 0U);

    assert(generic.pack_bits_lsb({}, {}) == Result::success);
    assert(generic.unpack_bits_lsb({}, {}) == Result::success);
    return 0;
}
