#pragma once

#include <cstdint>

#include <uni/simd/export.hpp>

namespace uni::simd {

struct Capabilities {
    bool sse2 = false;
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool avx512bw = false;
    bool neon = false;
};

[[nodiscard]] UNI_SIMD_API const Capabilities& capabilities() noexcept;

} // namespace uni::simd
