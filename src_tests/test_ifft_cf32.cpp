#ifdef NDEBUG
#undef NDEBUG
#endif

#include "common/api_internal.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

void compare_direct(const std::span<const float> input_real, const std::span<const float> input_imag,
                    const std::span<const float> actual_real, const std::span<const float> actual_imag) {
    const std::size_t count = input_real.size();
    for (std::size_t output = 0U; output < count; ++output) {
        std::complex<double> expected{};
        for (std::size_t input = 0U; input < count; ++input) {
            const double angle = 2.0 * pi * static_cast<double>(input * output) / static_cast<double>(count);
            expected += std::complex<double>{input_real[input], input_imag[input]} *
                        std::complex<double>{std::cos(angle), std::sin(angle)};
        }
        const float tolerance = 3.0e-5f + 2.0e-5f * static_cast<float>(std::abs(expected));
        assert(std::abs(actual_real[output] - static_cast<float>(expected.real())) <= tolerance);
        assert(std::abs(actual_imag[output] - static_cast<float>(expected.imag())) <= tolerance);
    }
}

void test_kernel(const uni::simd::IfftKernel& kernel) {
    const std::size_t count = kernel.size();
    for (std::size_t basis = 0U; basis < count; ++basis) {
        std::array<float, 32U> real{};
        std::array<float, 32U> imag{};
        real[basis] = 1.0f;
        assert(kernel.execute({.real = {real.data(), count}, .imag = {imag.data(), count}}) ==
               uni::simd::Result::success);
        for (std::size_t output = 0U; output < count; ++output) {
            const double angle = 2.0 * pi * static_cast<double>(basis * output) / static_cast<double>(count);
            assert(std::abs(real[output] - static_cast<float>(std::cos(angle))) < 2.0e-5f);
            assert(std::abs(imag[output] - static_cast<float>(std::sin(angle))) < 2.0e-5f);
        }

        real.fill(0.0f);
        imag.fill(0.0f);
        imag[basis] = 1.0f;
        assert(kernel.execute({.real = {real.data(), count}, .imag = {imag.data(), count}}) ==
               uni::simd::Result::success);
        for (std::size_t output = 0U; output < count; ++output) {
            const double angle = 2.0 * pi * static_cast<double>(basis * output) / static_cast<double>(count);
            assert(std::abs(real[output] + static_cast<float>(std::sin(angle))) < 2.0e-5f);
            assert(std::abs(imag[output] - static_cast<float>(std::cos(angle))) < 2.0e-5f);
        }
    }

    std::array<float, 32U> input_real{};
    std::array<float, 32U> input_imag{};
    for (std::size_t index = 0U; index < count; ++index) {
        input_real[index] = static_cast<float>(static_cast<std::int32_t>((index * 17U + 5U) % 23U) - 11) / 7.0f;
        input_imag[index] = static_cast<float>(static_cast<std::int32_t>((index * 11U + 3U) % 19U) - 9) / 5.0f;
    }
    auto actual_real = input_real;
    auto actual_imag = input_imag;
    assert(kernel.execute({.real = {actual_real.data(), count}, .imag = {actual_imag.data(), count}}) ==
           uni::simd::Result::success);
    compare_direct({input_real.data(), count}, {input_imag.data(), count},
                   {actual_real.data(), count}, {actual_imag.data(), count});

    constexpr std::size_t transform_count = 3U;
    const std::size_t stride = count + 3U;
    const std::size_t required = (transform_count - 1U) * stride + count;
    std::vector<float> batch_real(required, 1234.0f);
    std::vector<float> batch_imag(required, -1234.0f);
    for (std::size_t transform = 0U; transform < transform_count; ++transform) {
        for (std::size_t index = 0U; index < count; ++index) {
            batch_real[transform * stride + index] = input_real[index] + static_cast<float>(transform);
            batch_imag[transform * stride + index] = input_imag[index] - static_cast<float>(transform);
        }
    }
    const auto original_real = batch_real;
    const auto original_imag = batch_imag;
    assert(kernel.execute({.real = batch_real, .imag = batch_imag,
                           .transform_count = transform_count, .stride = stride}) ==
           uni::simd::Result::success);
    for (std::size_t transform = 0U; transform < transform_count; ++transform) {
        compare_direct({original_real.data() + transform * stride, count},
                       {original_imag.data() + transform * stride, count},
                       {batch_real.data() + transform * stride, count},
                       {batch_imag.data() + transform * stride, count});
        if (transform + 1U < transform_count) {
            for (std::size_t gap = count; gap < stride; ++gap) {
                assert(batch_real[transform * stride + gap] == 1234.0f);
                assert(batch_imag[transform * stride + gap] == -1234.0f);
            }
        }
    }

    std::vector<float> unaligned_real(count + 1U);
    std::vector<float> unaligned_imag(count + 1U);
    std::copy_n(input_real.begin(), count, unaligned_real.begin() + 1U);
    std::copy_n(input_imag.begin(), count, unaligned_imag.begin() + 1U);
    assert(kernel.execute({.real = {unaligned_real.data() + 1U, count},
                           .imag = {unaligned_imag.data() + 1U, count}}) ==
           uni::simd::Result::success);
    compare_direct({input_real.data(), count}, {input_imag.data(), count},
                   {unaligned_real.data() + 1U, count}, {unaligned_imag.data() + 1U, count});
}

} // namespace

int main() {
    std::set<std::pair<std::size_t, uni::simd::Backend>> tested_backends;
    for (const auto requested : {uni::simd::Backend::generic, uni::simd::Backend::avx2_fma,
                                  uni::simd::Backend::neon, uni::simd::Backend::automatic}) {
        const auto context = uni::simd::create_context({.backend = requested});
        if (!context) {
            assert(context.error() == uni::simd::Result::unsupported_backend);
            continue;
        }
        for (const std::size_t count : {4U, 8U, 16U, 32U}) {
            const auto kernel = context->make_ifft_cf32(count);
            assert(kernel.has_value());
            if ((requested == uni::simd::Backend::avx2_fma || requested == uni::simd::Backend::neon) &&
                count == 4U) {
                assert(kernel->backend() == uni::simd::Backend::generic);
            } else if (requested == uni::simd::Backend::avx2_fma || requested == uni::simd::Backend::neon) {
                assert(kernel->backend() == requested);
            }
            if (!tested_backends.emplace(count, kernel->backend()).second) {
                continue;
            }
            test_kernel(*kernel);
        }
    }

    const auto generic_context = *uni::simd::create_context({.backend = uni::simd::Backend::generic});
    const auto invalid_kernel = generic_context.make_ifft_cf32(3U);
    assert(!invalid_kernel.has_value() && invalid_kernel.error() == uni::simd::Result::invalid_size);
    const auto kernel = *generic_context.make_ifft_cf32(8U);
    std::array<float, 8U> real{};
    std::array<float, 7U> short_imag{};
    assert(kernel.execute({real, short_imag}) == uni::simd::Result::invalid_size);
    assert(kernel.execute({real, real}) == uni::simd::Result::overlapping_buffers);
    std::array<float, 16U> partially_overlapping{};
    assert(kernel.execute({.real = {partially_overlapping.data(), 8U},
                           .imag = {partially_overlapping.data() + 4U, 8U}}) ==
           uni::simd::Result::overlapping_buffers);
    assert(kernel.execute({.transform_count = 0U}) == uni::simd::Result::success);
    std::array<float, 16U> batch_real{};
    std::array<float, 16U> batch_imag{};
    assert(kernel.execute({.real = batch_real, .imag = batch_imag,
                           .transform_count = 2U, .stride = 7U}) == uni::simd::Result::invalid_size);
    assert(kernel.execute({.real = std::span<float>{batch_real}.first(15U), .imag = batch_imag,
                           .transform_count = 2U, .stride = 8U}) == uni::simd::Result::invalid_size);
    return 0;
}
