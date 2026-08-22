#ifdef NDEBUG
#undef NDEBUG
#endif

#include "common/api_internal.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <set>

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

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
    }
}

} // namespace

int main() {
    std::set<uni::simd::Backend> tested_eight_point_backends;
    for (const auto requested : {uni::simd::Backend::generic, uni::simd::Backend::avx2_fma,
                                 uni::simd::Backend::automatic}) {
        const auto context = uni::simd::create_context({.backend = requested});
        if (!context) {
            assert(context.error() == uni::simd::Result::unsupported_backend);
            continue;
        }
        for (const std::size_t count : {4U, 8U, 16U, 32U}) {
            const auto kernel = context->make_ifft_cf32(count);
            assert(kernel.has_value());
            if (count == 8U && requested == uni::simd::Backend::avx2_fma) {
                assert(kernel->backend() == uni::simd::Backend::avx2_fma);
            }
            if (count != 8U && requested == uni::simd::Backend::avx2_fma) {
                assert(kernel->backend() == uni::simd::Backend::generic);
            }
            if (count == 8U && !tested_eight_point_backends.insert(kernel->backend()).second) {
                continue;
            }
            test_kernel(*kernel);
        }
    }

    const auto generic_context = *uni::simd::create_context({.backend = uni::simd::Backend::generic});
    assert(!generic_context.make_ifft_cf32(3U).has_value());
    const auto kernel = *generic_context.make_ifft_cf32(8U);
    std::array<float, 8U> real{};
    std::array<float, 7U> short_imag{};
    assert(kernel.execute({real, short_imag}) == uni::simd::Result::invalid_size);
    assert(kernel.execute({real, real}) == uni::simd::Result::overlapping_buffers);
    std::array<float, 16U> partially_overlapping{};
    assert(kernel.execute({.real = {partially_overlapping.data(), 8U},
                           .imag = {partially_overlapping.data() + 4U, 8U}}) ==
           uni::simd::Result::overlapping_buffers);
    return 0;
}
