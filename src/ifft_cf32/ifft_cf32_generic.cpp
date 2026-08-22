#include "ifft_cf32/ifft_cf32_internal.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace uni::simd::detail {
namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

struct IfftTables final {
    std::array<float, 16U> twiddle_re{};
    std::array<float, 16U> twiddle_im{};
    std::array<std::array<std::uint8_t, 32U>, 4U> reverse{};

    IfftTables() noexcept {
        for (std::size_t index = 0U; index < twiddle_re.size(); ++index) {
            const double angle = 2.0 * pi * static_cast<double>(index) / 32.0;
            twiddle_re[index] = static_cast<float>(std::cos(angle));
            twiddle_im[index] = static_cast<float>(std::sin(angle));
        }
        for (std::size_t table = 0U, count = 4U; count <= 32U; ++table, count *= 2U) {
            std::size_t bits = 0U;
            for (std::size_t value = count; value > 1U; value /= 2U) {
                ++bits;
            }
            for (std::size_t value = 0U; value < count; ++value) {
                std::size_t source = value;
                std::size_t reversed = 0U;
                for (std::size_t bit = 0U; bit < bits; ++bit) {
                    reversed = (reversed << 1U) | (source & 1U);
                    source >>= 1U;
                }
                reverse[table][value] = static_cast<std::uint8_t>(reversed);
            }
        }
    }
};

const IfftTables tables;

[[nodiscard]] std::size_t table_index(const std::size_t count) noexcept {
    return count == 4U ? 0U : count == 8U ? 1U : count == 16U ? 2U : 3U;
}

} // namespace

void Ifft_generic(float* const real, float* const imag, const std::size_t count) noexcept {
    const auto& reverse = tables.reverse[table_index(count)];
    for (std::size_t index = 0U; index < count; ++index) {
        const std::size_t reversed = reverse[index];
        if (index < reversed) {
            std::swap(real[index], real[reversed]);
            std::swap(imag[index], imag[reversed]);
        }
    }

    for (std::size_t length = 2U; length <= count; length *= 2U) {
        const std::size_t half = length / 2U;
        const std::size_t twiddle_step = 32U / length;
        for (std::size_t base = 0U; base < count; base += length) {
            for (std::size_t index = 0U; index < half; ++index) {
                const std::size_t twiddle = index * twiddle_step;
                const std::size_t even = base + index;
                const std::size_t odd = even + half;
                const float odd_re = real[odd];
                const float odd_im = imag[odd];
                const float product_re = odd_re * tables.twiddle_re[twiddle] - odd_im * tables.twiddle_im[twiddle];
                const float product_im = odd_re * tables.twiddle_im[twiddle] + odd_im * tables.twiddle_re[twiddle];
                const float even_re = real[even];
                const float even_im = imag[even];
                real[even] = even_re + product_re;
                imag[even] = even_im + product_im;
                real[odd] = even_re - product_re;
                imag[odd] = even_im - product_im;
            }
        }
    }
}

} // namespace uni::simd::detail
