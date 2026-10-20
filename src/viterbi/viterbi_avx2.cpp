#include "viterbi_internal.hpp"

#if UNI_SIMD_HAVE_AVX2

#include <immintrin.h>

namespace uni::simd::kernels {

/**
 * Add-compare-select with the 64 16-bit path metrics held in four registers for the whole
 * block: states 0-15, 16-31, 32-47 and 48-63.
 *
 * A step's work is a dependency chain from one step's metrics to the next (add, min, and the
 * interleave that puts the new states back in order), so it is kept free of memory round trips
 * and of the per-step renormalisation of 8-bit decoders; renormalisation runs every
 * kViterbiRenormalizePeriod steps instead. Everything else (branch metrics, decisions) hangs
 * off that chain and overlaps with it.
 */
void ViterbiK7R2_avx2(ViterbiState& state, const std::uint8_t* const soft, const std::size_t steps, std::uint64_t* const decisions) noexcept {
    __m256i low0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state.metrics));
    __m256i low1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state.metrics + 16));
    __m256i high0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state.metrics + 32));
    __m256i high1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state.metrics + 48));
    const __m256i expected_a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state.expected_a));
    const __m256i expected_a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state.expected_a + 16));
    const __m256i expected_b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state.expected_b));
    const __m256i expected_b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(state.expected_b + 16));
    const __m256i one = _mm256_set1_epi16(1);
    const __m256i max_metric = _mm256_set1_epi16(63);
    std::size_t since_renormalize = state.steps_since_renormalize;

    for (std::size_t step = 0U; step < steps; ++step) {
        const __m256i s0 = _mm256_set1_epi16(soft[2U * step]);
        const __m256i s1 = _mm256_set1_epi16(soft[2U * step + 1U]);
        // Branch metric of old state i with a new 0 bit, butterflies 0-15 and 16-31.
        const auto branch = [&](const __m256i a, const __m256i b) noexcept {
            return _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_xor_si256(a, s0), _mm256_xor_si256(b, s1)), one), 3);
        };
        const __m256i metric0 = branch(expected_a0, expected_b0);
        const __m256i metric1 = branch(expected_a1, expected_b1);
        const __m256i complement0 = _mm256_sub_epi16(max_metric, metric0);
        const __m256i complement1 = _mm256_sub_epi16(max_metric, metric1);

        // Butterfly i: old states i and i + 32 -> new states 2i (even) and 2i + 1 (odd).
        const auto butterflies = [&](const __m256i low, const __m256i high, const __m256i metric, const __m256i complement, __m256i& next0, __m256i& next1,
                                     std::uint32_t& decided) noexcept {
            const __m256i from_high_even = _mm256_add_epi16(high, complement);
            const __m256i from_high_odd = _mm256_add_epi16(high, metric);
            const __m256i even = _mm256_min_epu16(_mm256_add_epi16(low, metric), from_high_even);
            const __m256i odd = _mm256_min_epu16(_mm256_add_epi16(low, complement), from_high_odd);
            // Ties go to the path from the high state, as in VOLK.
            const __m256i even_high = _mm256_cmpeq_epi16(even, from_high_even);
            const __m256i odd_high = _mm256_cmpeq_epi16(odd, from_high_odd);
            // Interleave to state order: per 128-bit lane the unpacks give states 0-7 / 8-15 of
            // the lower eight butterflies and 16-23 / 24-31 of the upper ones.
            const __m256i interleaved_low = _mm256_unpacklo_epi16(even, odd);
            const __m256i interleaved_high = _mm256_unpackhi_epi16(even, odd);
            next0 = _mm256_permute2x128_si256(interleaved_low, interleaved_high, 0x20);
            next1 = _mm256_permute2x128_si256(interleaved_low, interleaved_high, 0x31);
            // Packing the interleaved decisions to bytes restores state order within each lane.
            decided = static_cast<std::uint32_t>(_mm256_movemask_epi8(
                _mm256_packs_epi16(_mm256_unpacklo_epi16(even_high, odd_high), _mm256_unpackhi_epi16(even_high, odd_high))));
        };
        std::uint32_t decided_low = 0U;
        std::uint32_t decided_high = 0U;
        __m256i next_low0;
        __m256i next_low1;
        __m256i next_high0;
        __m256i next_high1;
        butterflies(low0, high0, metric0, complement0, next_low0, next_low1, decided_low);
        butterflies(low1, high1, metric1, complement1, next_high0, next_high1, decided_high);
        decisions[step] = static_cast<std::uint64_t>(decided_low) | (static_cast<std::uint64_t>(decided_high) << 32U);
        low0 = next_low0;
        low1 = next_low1;
        high0 = next_high0;
        high1 = next_high1;

        if (++since_renormalize == kViterbiRenormalizePeriod) {
            const __m256i minimum4 = _mm256_min_epu16(_mm256_min_epu16(low0, low1), _mm256_min_epu16(high0, high1));
            const __m128i minimum8 = _mm_min_epu16(_mm256_castsi256_si128(minimum4), _mm256_extracti128_si256(minimum4, 1));
            const __m256i minimum = _mm256_set1_epi16(static_cast<short>(_mm_extract_epi16(_mm_minpos_epu16(minimum8), 0)));
            low0 = _mm256_sub_epi16(low0, minimum);
            low1 = _mm256_sub_epi16(low1, minimum);
            high0 = _mm256_sub_epi16(high0, minimum);
            high1 = _mm256_sub_epi16(high1, minimum);
            since_renormalize = 0U;
        }
    }

    _mm256_store_si256(reinterpret_cast<__m256i*>(state.metrics), low0);
    _mm256_store_si256(reinterpret_cast<__m256i*>(state.metrics + 16), low1);
    _mm256_store_si256(reinterpret_cast<__m256i*>(state.metrics + 32), high0);
    _mm256_store_si256(reinterpret_cast<__m256i*>(state.metrics + 48), high1);
    state.steps_since_renormalize = since_renormalize;
}

} // namespace uni::simd::kernels

#endif
