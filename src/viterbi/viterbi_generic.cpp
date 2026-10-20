#include "viterbi_internal.hpp"

namespace uni::simd::kernels {

void ViterbiK7R2_generic(ViterbiState& state, const std::uint8_t* const soft, const std::size_t steps, std::uint64_t* const decisions) noexcept {
    std::uint16_t next[kViterbiStates];
    for (std::size_t step = 0U; step < steps; ++step) {
        const std::uint16_t s0 = soft[2U * step];
        const std::uint16_t s1 = soft[2U * step + 1U];
        std::uint64_t word = 0U;
        for (std::size_t butterfly = 0U; butterfly < kViterbiButterflies; ++butterfly) {
            // Cost of old state i taking a new 0 bit; the other three branches of the butterfly
            // expect the complementary bits (the code polynomials tap the newest and oldest bit).
            const auto metric = static_cast<std::uint16_t>(((state.expected_a[butterfly] ^ s0) + (state.expected_b[butterfly] ^ s1) + 1U) >> 3U);
            const auto complement = static_cast<std::uint16_t>(63U - metric);
            const std::uint16_t low = state.metrics[butterfly];
            const std::uint16_t high = state.metrics[butterfly + kViterbiButterflies];
            const auto m0 = static_cast<std::uint16_t>(low + metric);
            const auto m1 = static_cast<std::uint16_t>(high + complement);
            const auto m2 = static_cast<std::uint16_t>(low + complement);
            const auto m3 = static_cast<std::uint16_t>(high + metric);
            next[2U * butterfly] = m1 <= m0 ? m1 : m0;
            next[2U * butterfly + 1U] = m3 <= m2 ? m3 : m2;
            word |= static_cast<std::uint64_t>(m1 <= m0) << (2U * butterfly);
            word |= static_cast<std::uint64_t>(m3 <= m2) << (2U * butterfly + 1U);
        }
        decisions[step] = word;
        for (std::size_t index = 0U; index < kViterbiStates; ++index) {
            state.metrics[index] = next[index];
        }
        if (++state.steps_since_renormalize == kViterbiRenormalizePeriod) {
            ViterbiRenormalize(state);
        }
    }
}

} // namespace uni::simd::kernels
