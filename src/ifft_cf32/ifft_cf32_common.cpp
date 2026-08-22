#include "common/api_internal.hpp"

#include <cstddef>
#include <cstdint>

namespace uni::simd {

Result IfftKernel::execute(const IfftSplitComplex values) const noexcept {
    if (function_ == nullptr) {
        return Result::invalid_argument;
    }
    if (values.real.size() != size_ || values.imag.size() != size_) {
        return Result::invalid_size;
    }
    const auto real_begin = reinterpret_cast<std::uintptr_t>(values.real.data());
    const auto imag_begin = reinterpret_cast<std::uintptr_t>(values.imag.data());
    if (real_begin < imag_begin + values.imag.size_bytes() &&
        imag_begin < real_begin + values.real.size_bytes()) {
        return Result::overlapping_buffers;
    }
    function_(values.real.data(), values.imag.data(), size_);
    return Result::success;
}

} // namespace uni::simd
