#include <uni/simd/capabilities.hpp>

#include <cstdint>

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif
#endif

namespace uni::simd {
namespace {

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
struct CpuidRegs {
    std::uint32_t eax{};
    std::uint32_t ebx{};
    std::uint32_t ecx{};
    std::uint32_t edx{};
};

[[nodiscard]] CpuidRegs cpuid(const std::uint32_t leaf, const std::uint32_t subleaf) noexcept {
    CpuidRegs result{};
#if defined(_MSC_VER)
    int registers[4]{};
    __cpuidex(registers, static_cast<int>(leaf), static_cast<int>(subleaf));
    result = {static_cast<std::uint32_t>(registers[0]), static_cast<std::uint32_t>(registers[1]), static_cast<std::uint32_t>(registers[2]),
              static_cast<std::uint32_t>(registers[3])};
#elif defined(__GNUC__) || defined(__clang__)
    __cpuid_count(leaf, subleaf, result.eax, result.ebx, result.ecx, result.edx);
#endif
    return result;
}

[[nodiscard]] std::uint64_t xgetbv(const std::uint32_t index) noexcept {
#if defined(_MSC_VER)
    return _xgetbv(index);
#else
    std::uint32_t eax{};
    std::uint32_t edx{};
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return (static_cast<std::uint64_t>(edx) << 32U) | eax;
#endif
}
#endif

[[nodiscard]] Capabilities detect_capabilities() noexcept {
    Capabilities result{};
#if defined(__aarch64__) || defined(_M_ARM64)
    result.neon = true;
#elif defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    constexpr std::uint32_t sse2_bit = 1U << 26U;
    constexpr std::uint32_t fma_bit = 1U << 12U;
    constexpr std::uint32_t osxsave_bit = 1U << 27U;
    constexpr std::uint32_t avx_bit = 1U << 28U;
    constexpr std::uint32_t avx2_bit = 1U << 5U;
    constexpr std::uint32_t avx512f_bit = 1U << 16U;
    constexpr std::uint32_t avx512bw_bit = 1U << 30U;
    constexpr std::uint64_t avx_state = 0x6U;
    constexpr std::uint64_t avx512_state = avx_state | 0xE0U;

    const auto max_leaf = cpuid(0U, 0U).eax;
    if (max_leaf >= 1U) {
        const auto leaf1 = cpuid(1U, 0U);
        result.sse2 = (leaf1.edx & sse2_bit) != 0U;
        const bool has_osxsave = (leaf1.ecx & osxsave_bit) != 0U;
        const bool has_avx = (leaf1.ecx & avx_bit) != 0U;
        const auto xcr0 = has_osxsave ? xgetbv(0U) : 0U;
        const bool avx_os = has_avx && (xcr0 & avx_state) == avx_state;
        result.fma = avx_os && (leaf1.ecx & fma_bit) != 0U;

        if (max_leaf >= 7U && avx_os) {
            const auto leaf7 = cpuid(7U, 0U);
            result.avx2 = (leaf7.ebx & avx2_bit) != 0U;
            const bool avx512_os = (xcr0 & avx512_state) == avx512_state;
            result.avx512f = avx512_os && (leaf7.ebx & avx512f_bit) != 0U;
            result.avx512bw = result.avx512f && (leaf7.ebx & avx512bw_bit) != 0U;
        }
    }
#endif
    return result;
}

} // namespace

const Capabilities& capabilities() noexcept {
    static const Capabilities detected = detect_capabilities();
    return detected;
}

} // namespace uni::simd
