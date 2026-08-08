#pragma once

#include <cstdint>

namespace uni::simd {

enum class Result : std::uint8_t {
    success,
    invalid_argument,
    invalid_size,
    overlapping_buffers,
    unsupported_backend,
};

[[nodiscard]] constexpr bool succeeded(const Result result) noexcept {
    return result == Result::success;
}

} // namespace uni::simd
