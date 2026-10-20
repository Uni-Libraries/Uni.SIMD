#include "uni_simd.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

void Check(const bool condition, const char* const message) {
    if (!condition) {
        std::fprintf(stderr, "viterbi_k7r2 test failed: %s\n", message);
        std::abort();
    }
}

constexpr std::uint32_t kPolynomialA = 0171U;
constexpr std::uint32_t kPolynomialB = 0133U;
constexpr std::uint32_t kStartPenalty = 31U;

[[nodiscard]] std::uint8_t Parity(const std::uint32_t value) noexcept { return static_cast<std::uint8_t>(std::popcount(value) & 1); }

/** CCSDS-style encoder: the newest bit enters at bit 6 of the register. */
[[nodiscard]] std::vector<std::uint8_t> Encode(const std::vector<std::uint8_t>& bits, const double noise, const unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> gaussian(0.0, noise);
    std::vector<std::uint8_t> soft;
    std::uint32_t memory = 0U;
    for (const auto bit : bits) {
        const std::uint32_t shift = (static_cast<std::uint32_t>(bit) << 6U) | memory;
        for (const auto polynomial : {kPolynomialA, kPolynomialB}) {
            const double value = (Parity(shift & polynomial) != 0U ? 200.0 : 55.0) + (noise > 0.0 ? gaussian(rng) : 0.0);
            soft.push_back(static_cast<std::uint8_t>(std::clamp(value, 0.0, 255.0)));
        }
        memory = (shift >> 1U) & 0x3FU;
    }
    return soft;
}

/** Straightforward reference: 32-bit metrics, same branch metric and tie rule, no renormalisation. */
[[nodiscard]] std::vector<std::uint64_t> ReferenceDecisions(const std::vector<std::uint8_t>& soft, std::uint32_t& best_state) {
    std::array<std::uint32_t, 64U> metrics{};
    metrics.fill(kStartPenalty);
    metrics[0] = 0U;
    const auto expected = [](const std::uint32_t polynomial, const std::uint32_t butterfly) {
        std::uint32_t register_bits = 0U;
        for (std::uint32_t bit = 0U; bit < 7U; ++bit) {
            register_bits |= (((butterfly << 1U) >> bit) & 1U) << (6U - bit);
        }
        return Parity(polynomial & register_bits) != 0U ? 255U : 0U;
    };
    std::vector<std::uint64_t> decisions;
    for (std::size_t step = 0U; step < soft.size() / 2U; ++step) {
        std::array<std::uint32_t, 64U> next{};
        std::uint64_t word = 0U;
        for (std::uint32_t butterfly = 0U; butterfly < 32U; ++butterfly) {
            const std::uint32_t metric = ((expected(kPolynomialA, butterfly) ^ soft[2U * step]) + (expected(kPolynomialB, butterfly) ^ soft[2U * step + 1U]) + 1U) >> 3U;
            const std::uint32_t m0 = metrics[butterfly] + metric;
            const std::uint32_t m1 = metrics[butterfly + 32U] + (63U - metric);
            const std::uint32_t m2 = metrics[butterfly] + (63U - metric);
            const std::uint32_t m3 = metrics[butterfly + 32U] + metric;
            next[2U * butterfly] = std::min(m0, m1);
            next[2U * butterfly + 1U] = std::min(m2, m3);
            word |= static_cast<std::uint64_t>(m1 <= m0) << (2U * butterfly);
            word |= static_cast<std::uint64_t>(m3 <= m2) << (2U * butterfly + 1U);
        }
        metrics = next;
        decisions.push_back(word);
    }
    best_state = static_cast<std::uint32_t>(std::min_element(metrics.begin(), metrics.end()) - metrics.begin());
    return decisions;
}

struct Run {
    std::vector<std::uint64_t> decisions;
    std::uint32_t best_state{};
};

[[nodiscard]] Run Decode(const std::vector<std::uint8_t>& soft, const uni_simd_backend_e backend, const std::vector<std::size_t>& splits) {
    uni_simd_viterbi_k7r2_config_t config{};
    config.descriptor_size = UNI_SIMD_VITERBI_K7R2_CONFIG_DESCRIPTOR_SIZE;
    config.polynomial_a = kPolynomialA;
    config.polynomial_b = kPolynomialB;
    config.known_start_state = 1U;
    config.start_penalty = kStartPenalty;
    auto* const kernel = uni_simd_kernel_create(UNI_SIMD_KERNEL_VITERBI_K7R2_U8);
    Check(kernel != nullptr, "create");
    std::array params{uni_simd_param_t{UNI_SIMD_PARAM_CONFIG, {.const_pointer = &config}}, uni_simd_param_t{UNI_SIMD_PARAM_BACKEND, {.u32 = backend}}};
    Check(uni_simd_kernel_param_set_many(kernel, params.data(), params.size()) == UNI_SIMD_RESULT_SUCCESS, "params");

    Run run;
    const std::size_t steps = soft.size() / 2U;
    run.decisions.resize(steps);
    std::size_t done = 0U;
    for (std::size_t turn = 0U; done < steps; ++turn) {
        const std::size_t count = std::min(splits[turn % splits.size()], steps - done);
        uni_simd_viterbi_k7r2_block_t block{UNI_SIMD_VITERBI_K7R2_BLOCK_DESCRIPTOR_SIZE, soft.data() + 2U * done, count, run.decisions.data() + done};
        uni_simd_viterbi_k7r2_result_t result{};
        result.descriptor_size = UNI_SIMD_VITERBI_K7R2_RESULT_DESCRIPTOR_SIZE;
        Check(uni_simd_kernel_execute(kernel, &block, &result) == UNI_SIMD_RESULT_SUCCESS, "execute");
        run.best_state = result.best_state;
        done += count;
    }
    Check(uni_simd_kernel_free(kernel) == UNI_SIMD_RESULT_SUCCESS, "free");
    return run;
}

/** Traceback over the whole stream from `state` (newest bit in bit 0 of a state). */
[[nodiscard]] std::vector<std::uint8_t> Traceback(const std::vector<std::uint64_t>& decisions, std::uint32_t state) {
    std::vector<std::uint8_t> bits(decisions.size());
    for (std::size_t index = decisions.size(); index > 0U; --index) {
        bits[index - 1U] = static_cast<std::uint8_t>(state & 1U);
        state = ((state >> 1U) | (static_cast<std::uint32_t>((decisions[index - 1U] >> state) & 1U) << 5U)) & 0x3FU;
    }
    return bits;
}

} // namespace

int main() {
    Check(uni_simd_initialize() == UNI_SIMD_RESULT_SUCCESS, "initialize");
    std::mt19937 rng(7);
    std::vector<std::uint8_t> bits(200000U);
    for (auto& bit : bits) {
        bit = static_cast<std::uint8_t>(rng() & 1U);
    }

    for (const double noise : {0.0, 40.0, 90.0}) {
        const auto soft = Encode(bits, noise, 11U);
        std::uint32_t reference_best = 0U;
        const auto reference = ReferenceDecisions(soft, reference_best);
        for (const auto backend : {UNI_SIMD_BACKEND_GENERIC, UNI_SIMD_BACKEND_AUTOMATIC}) {
            const auto whole = Decode(soft, backend, {soft.size()});
            Check(whole.decisions == reference, "decisions match the reference");
            Check(whole.best_state == reference_best, "best state matches the reference");
            const auto split = Decode(soft, backend, {1U, 255U, 1000U, 3U, 4096U});
            Check(split.decisions == reference && split.best_state == reference_best, "decisions do not depend on the block split");
        }
        if (noise <= 40.0) {
            const auto decoded = Traceback(reference, reference_best);
            Check(decoded == bits, "a clean or moderately noisy stream decodes without errors");
        }
    }

    uni_simd_viterbi_k7r2_config_t bad{};
    bad.descriptor_size = UNI_SIMD_VITERBI_K7R2_CONFIG_DESCRIPTOR_SIZE;
    bad.polynomial_a = 0x80U;
    bad.polynomial_b = kPolynomialB;
    auto* const kernel = uni_simd_kernel_create(UNI_SIMD_KERNEL_VITERBI_K7R2_U8);
    Check(uni_simd_kernel_param_set(kernel, {UNI_SIMD_PARAM_CONFIG, {.const_pointer = &bad}}) == UNI_SIMD_RESULT_SUCCESS, "params");
    uni_simd_viterbi_k7r2_block_t block{UNI_SIMD_VITERBI_K7R2_BLOCK_DESCRIPTOR_SIZE, nullptr, 0U, nullptr};
    uni_simd_viterbi_k7r2_result_t result{};
    result.descriptor_size = UNI_SIMD_VITERBI_K7R2_RESULT_DESCRIPTOR_SIZE;
    Check(uni_simd_kernel_execute(kernel, &block, &result) == UNI_SIMD_RESULT_INVALID_ARGUMENT, "an 8-bit polynomial is rejected");
    Check(uni_simd_kernel_free(kernel) == UNI_SIMD_RESULT_SUCCESS, "free");

    Check(uni_simd_finalize() == UNI_SIMD_RESULT_SUCCESS, "finalize");
    std::puts("viterbi_k7r2 tests passed");
    return 0;
}
