#include "common/api_internal.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace uni::simd {

Result IfftKernel::execute(const IfftSplitComplex values) const noexcept {
    if (function_ == nullptr) {
        return Result::invalid_argument;
    }
    const std::size_t stride = values.stride == 0U ? size_ : values.stride;
    if (values.transform_count == 0U) {
        return Result::success;
    }
    if (stride < size_ || values.transform_count - 1U >
                              (std::numeric_limits<std::size_t>::max() - size_) / stride) {
        return Result::invalid_size;
    }
    const std::size_t required = (values.transform_count - 1U) * stride + size_;
    if (required > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        values.real.size() < required || values.imag.size() < required) {
        return Result::invalid_size;
    }
    const auto real_begin = reinterpret_cast<std::uintptr_t>(values.real.data());
    const auto imag_begin = reinterpret_cast<std::uintptr_t>(values.imag.data());
    const std::size_t required_bytes = required * sizeof(float);
    if (real_begin <= imag_begin ? imag_begin - real_begin < required_bytes
                                 : real_begin - imag_begin < required_bytes) {
        return Result::overlapping_buffers;
    }
    function_(values.real.data(), values.imag.data(), size_, values.transform_count, stride);
    return Result::success;
}

} // namespace uni::simd
