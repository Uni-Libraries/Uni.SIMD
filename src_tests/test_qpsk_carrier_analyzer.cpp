#include "uni_simd.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numbers>
#include <vector>

namespace {

void Check(const bool condition, const char* const message) {
    if (!condition) {
        std::fprintf(stderr, "qpsk carrier analyzer test failed: %s\n", message);
        std::abort();
    }
}

void CheckClose(const float actual, const float expected, const float tolerance, const char* const message) {
    if (std::abs(actual - expected) > tolerance + tolerance * std::abs(expected)) {
        std::fprintf(stderr, "qpsk carrier analyzer test failed: %s: %.9g != %.9g\n", message, static_cast<double>(actual), static_cast<double>(expected));
        std::abort();
    }
}

[[nodiscard]] uni_simd_kernel_t* Create(const uni_simd_backend_e backend, const uni_simd_math_mode_e math_mode = UNI_SIMD_MATH_FAST) {
    uni_simd_qpsk_carrier_analyzer_config_t config{};
    config.descriptor_size = UNI_SIMD_QPSK_CARRIER_ANALYZER_CONFIG_DESCRIPTOR_SIZE;
    config.magnitude_epsilon = 1.0e-12f;
    auto* const analyzer = uni_simd_kernel_create(UNI_SIMD_KERNEL_QPSK_CARRIER_ANALYZER_CF32);
    Check(analyzer != nullptr, "created analyzer");
    std::array params{
        uni_simd_param_t{UNI_SIMD_PARAM_CONFIG, {.const_pointer = &config}},
        uni_simd_param_t{UNI_SIMD_PARAM_BACKEND, {.u32 = backend}},
        uni_simd_param_t{UNI_SIMD_PARAM_MATH_MODE, {.u32 = math_mode}},
    };
    Check(uni_simd_kernel_param_set_many(analyzer, params.data(), params.size()) == UNI_SIMD_RESULT_SUCCESS, "configure");
    const uni_simd_qpsk_carrier_analyzer_block_t empty{UNI_SIMD_QPSK_CARRIER_ANALYZER_BLOCK_DESCRIPTOR_SIZE, nullptr, 0U};
    uni_simd_qpsk_carrier_analyzer_result_t result{};
    Check(uni_simd_kernel_execute(analyzer, &empty, &result) == UNI_SIMD_RESULT_SUCCESS, "initialize state");
    return analyzer;
}

[[nodiscard]] uni_simd_result_e TryCreate(const uni_simd_backend_e backend, uni_simd_kernel_t** const analyzer) {
    uni_simd_qpsk_carrier_analyzer_config_t config{};
    config.descriptor_size = UNI_SIMD_QPSK_CARRIER_ANALYZER_CONFIG_DESCRIPTOR_SIZE;
    config.magnitude_epsilon = 1.0e-12f;
    *analyzer = uni_simd_kernel_create(UNI_SIMD_KERNEL_QPSK_CARRIER_ANALYZER_CF32);
    if (*analyzer == nullptr) {
        return UNI_SIMD_RESULT_NOT_INITIALIZED;
    }
    std::array params{
        uni_simd_param_t{UNI_SIMD_PARAM_CONFIG, {.const_pointer = &config}},
        uni_simd_param_t{UNI_SIMD_PARAM_BACKEND, {.u32 = backend}},
    };
    const auto configured = uni_simd_kernel_param_set_many(*analyzer, params.data(), params.size());
    if (configured != UNI_SIMD_RESULT_SUCCESS) {
        (void)uni_simd_kernel_free(*analyzer);
        *analyzer = nullptr;
        return configured;
    }
    const uni_simd_qpsk_carrier_analyzer_block_t empty{UNI_SIMD_QPSK_CARRIER_ANALYZER_BLOCK_DESCRIPTOR_SIZE, nullptr, 0U};
    uni_simd_qpsk_carrier_analyzer_result_t result{};
    const auto status = uni_simd_kernel_execute(*analyzer, &empty, &result);
    if (status != UNI_SIMD_RESULT_SUCCESS) {
        (void)uni_simd_kernel_free(*analyzer);
        *analyzer = nullptr;
    }
    return status;
}

[[nodiscard]] uni_simd_qpsk_carrier_analyzer_result_t Analyze(uni_simd_kernel_t* const analyzer, const float* const samples,
                                                              const std::size_t count) {
    uni_simd_qpsk_carrier_analyzer_block_t block{};
    block.descriptor_size = UNI_SIMD_QPSK_CARRIER_ANALYZER_BLOCK_DESCRIPTOR_SIZE;
    block.samples = samples;
    block.sample_count = count;
    uni_simd_qpsk_carrier_analyzer_result_t result{};
    Check(uni_simd_kernel_execute(analyzer, &block, &result) == UNI_SIMD_RESULT_SUCCESS, "analyze");
    Check(result.descriptor_size == UNI_SIMD_QPSK_CARRIER_ANALYZER_RESULT_DESCRIPTOR_SIZE, "result descriptor");
    return result;
}

[[nodiscard]] uni_simd_backend_e Backend(uni_simd_kernel_t* const analyzer) {
    uni_simd_backend_e backend = UNI_SIMD_BACKEND_AUTOMATIC;
    Check(uni_simd_kernel_param_set(analyzer, {UNI_SIMD_PARAM_RESOLVED_BACKEND, {.pointer = &backend}}) == UNI_SIMD_RESULT_SUCCESS,
          "backend output");
    const uni_simd_qpsk_carrier_analyzer_block_t empty{UNI_SIMD_QPSK_CARRIER_ANALYZER_BLOCK_DESCRIPTOR_SIZE, nullptr, 0U};
    uni_simd_qpsk_carrier_analyzer_result_t result{};
    Check(uni_simd_kernel_execute(analyzer, &empty, &result) == UNI_SIMD_RESULT_SUCCESS, "backend query");
    Check(uni_simd_kernel_param_set(analyzer, {UNI_SIMD_PARAM_RESOLVED_BACKEND, {.pointer = nullptr}}) == UNI_SIMD_RESULT_SUCCESS,
          "clear backend output");
    return backend;
}

void Compare(const uni_simd_qpsk_carrier_analyzer_result_t& actual, const uni_simd_qpsk_carrier_analyzer_result_t& expected, const float tolerance) {
    for (std::size_t component = 0U; component < 2U; ++component) {
        CheckClose(actual.fourth_sum[component], expected.fourth_sum[component], tolerance, "fourth sum");
        CheckClose(actual.adjacent_fourth_sum[component], expected.adjacent_fourth_sum[component], tolerance, "adjacent fourth sum");
        CheckClose(actual.decision_sum[component], expected.decision_sum[component], tolerance, "decision sum");
    }
    CheckClose(actual.input_power, expected.input_power, tolerance, "input power");
    CheckClose(actual.valid_fourth_weight, expected.valid_fourth_weight, tolerance, "valid fourth weight");
    Check(actual.valid_fourth_count == expected.valid_fourth_count, "valid count");
    Check(actual.adjacent_fourth_count == expected.adjacent_fourth_count, "adjacent count");
}

void Add(uni_simd_qpsk_carrier_analyzer_result_t& destination, const uni_simd_qpsk_carrier_analyzer_result_t& source) {
    for (std::size_t component = 0U; component < 2U; ++component) {
        destination.fourth_sum[component] += source.fourth_sum[component];
        destination.adjacent_fourth_sum[component] += source.adjacent_fourth_sum[component];
        destination.decision_sum[component] += source.decision_sum[component];
    }
    destination.input_power += source.input_power;
    destination.valid_fourth_weight += source.valid_fourth_weight;
    destination.valid_fourth_count += source.valid_fourth_count;
    destination.adjacent_fourth_count += source.adjacent_fourth_count;
}

[[nodiscard]] std::vector<float> AnalyticQpsk(const std::size_t count, const float phase, const float omega, const float amplitude) {
    constexpr std::array<std::array<float, 2U>, 4U> symbols{{
        {{1.0f, 1.0f}},
        {{-1.0f, 1.0f}},
        {{-1.0f, -1.0f}},
        {{1.0f, -1.0f}},
    }};
    std::vector<float> result(2U * count);
    for (std::size_t index = 0U; index < count; ++index) {
        const float angle = phase + omega * static_cast<float>(index);
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const auto symbol = symbols[index % symbols.size()];
        result[2U * index] = amplitude * (symbol[0] * cosine - symbol[1] * sine);
        result[2U * index + 1U] = amplitude * (symbol[0] * sine + symbol[1] * cosine);
    }
    return result;
}

void TestAnalyticSigns() {
    constexpr std::size_t count = 17U;
    constexpr float phase = 0.03f;
    constexpr float omega = 0.011f;
    constexpr float amplitude = 0.75f;
    const auto input = AnalyticQpsk(count, phase, omega, amplitude);
    auto* const analyzer = Create(UNI_SIMD_BACKEND_GENERIC);
    const auto result = Analyze(analyzer, input.data(), count);

    float expected_fourth_real = 0.0f;
    float expected_fourth_imag = 0.0f;
    float expected_decision_real = 0.0f;
    float expected_decision_imag = 0.0f;
    for (std::size_t index = 0U; index < count; ++index) {
        const float angle = phase + omega * static_cast<float>(index);
        expected_fourth_real -= std::cos(4.0f * angle);
        expected_fourth_imag -= std::sin(4.0f * angle);
        expected_decision_real += 2.0f * amplitude * std::cos(angle);
        expected_decision_imag += 2.0f * amplitude * std::sin(angle);
    }
    CheckClose(result.fourth_sum[0], expected_fourth_real, 2.0e-5f, "analytic fourth real");
    CheckClose(result.fourth_sum[1], expected_fourth_imag, 2.0e-5f, "analytic fourth imag");
    CheckClose(result.adjacent_fourth_sum[0], static_cast<float>(count - 1U) * std::cos(4.0f * omega), 2.0e-5f, "analytic adjacent real");
    CheckClose(result.adjacent_fourth_sum[1], static_cast<float>(count - 1U) * std::sin(4.0f * omega), 2.0e-5f, "analytic adjacent positive CFO sign");
    CheckClose(result.decision_sum[0], expected_decision_real, 2.0e-5f, "analytic decision real");
    CheckClose(result.decision_sum[1], expected_decision_imag, 2.0e-5f, "analytic decision positive phase sign");
    Check(result.adjacent_fourth_sum[1] > 0.0f, "positive CFO sign");
    Check(result.decision_sum[1] > 0.0f, "positive phase sign");
    CheckClose(result.input_power, static_cast<float>(count) * 2.0f * amplitude * amplitude, 2.0e-6f, "analytic input power");
    CheckClose(result.valid_fourth_weight, result.input_power, 1.0e-6f, "analytic valid weight");
    Check(result.valid_fourth_count == count, "analytic valid count");
    Check(result.adjacent_fourth_count == count - 1U, "analytic adjacent count");
    Check(uni_simd_kernel_free(analyzer) == UNI_SIMD_RESULT_SUCCESS, "free");
}

void TestStreamingResetAndZeros() {
    const auto input = AnalyticQpsk(13U, 0.02f, -0.007f, 0.6f);
    auto* const whole_analyzer = Create(UNI_SIMD_BACKEND_GENERIC);
    const auto whole = Analyze(whole_analyzer, input.data(), 13U);

    auto* const split_analyzer = Create(UNI_SIMD_BACKEND_GENERIC);
    auto combined = Analyze(split_analyzer, input.data(), 5U);
    const auto empty = Analyze(split_analyzer, nullptr, 0U);
    Check(empty.valid_fourth_count == 0U && empty.adjacent_fourth_count == 0U, "empty result");
    const auto second = Analyze(split_analyzer, input.data() + 10U, 8U);
    Add(combined, second);
    Compare(combined, whole, 2.0e-6f);

    Check(uni_simd_kernel_reset(split_analyzer) == UNI_SIMD_RESULT_SUCCESS, "reset");
    const auto after_reset = Analyze(split_analyzer, input.data() + 10U, 8U);
    Check(after_reset.adjacent_fourth_count == 7U, "reset removes boundary pair");

    const std::array<float, 6U> with_zero{1.0f, 1.0f, 0.0f, 0.0f, -1.0f, 1.0f};
    Check(uni_simd_kernel_reset(split_analyzer) == UNI_SIMD_RESULT_SUCCESS, "zero reset");
    const auto zero_result = Analyze(split_analyzer, with_zero.data(), 3U);
    Check(zero_result.valid_fourth_count == 2U, "zero valid count");
    Check(zero_result.adjacent_fourth_count == 0U, "zero breaks adjacency");

    Check(uni_simd_kernel_free(whole_analyzer) == UNI_SIMD_RESULT_SUCCESS, "free whole");
    Check(uni_simd_kernel_free(split_analyzer) == UNI_SIMD_RESULT_SUCCESS, "free split");
}

void TestBackendsOddAndUnaligned() {
    constexpr std::size_t count = 1031U;
    const auto generated = AnalyticQpsk(count, -0.12f, 0.0007f, 0.83f);
    std::vector<float> storage(generated.size() + 1U);
    std::copy(generated.begin(), generated.end(), storage.begin() + 1U);
    const float* const unaligned = storage.data() + 1U;

    auto* const generic = Create(UNI_SIMD_BACKEND_GENERIC);
    const auto expected = Analyze(generic, unaligned, count);
    Check(Backend(generic) == UNI_SIMD_BACKEND_GENERIC, "generic backend report");

    std::vector<uni_simd_backend_e> resolved;
    for (const auto requested : {UNI_SIMD_BACKEND_X86_AVX2, UNI_SIMD_BACKEND_X86_AVX2_FMA, UNI_SIMD_BACKEND_X86_AVX512, UNI_SIMD_BACKEND_AARCH64_NEON}) {
        uni_simd_kernel_t* analyzer = nullptr;
        const auto status = TryCreate(requested, &analyzer);
        if (status == UNI_SIMD_RESULT_UNSUPPORTED_BACKEND) {
            Check(analyzer == nullptr, "unsupported backend null result");
            continue;
        }
        Check(status == UNI_SIMD_RESULT_SUCCESS && analyzer != nullptr, "forced backend create");
        const auto actual_backend = Backend(analyzer);
        Check(actual_backend != UNI_SIMD_BACKEND_AUTOMATIC && actual_backend != UNI_SIMD_BACKEND_GENERIC, "forced backend report");
        if (requested == UNI_SIMD_BACKEND_X86_AVX2_FMA) {
            Check(actual_backend == UNI_SIMD_BACKEND_X86_AVX2_FMA, "AVX2/FMA report");
        } else if (requested == UNI_SIMD_BACKEND_X86_AVX512) {
            Check(actual_backend == UNI_SIMD_BACKEND_X86_AVX512 || actual_backend == UNI_SIMD_BACKEND_X86_AVX2_FMA, "AVX-512 report or fallback");
        } else {
            Check(actual_backend == UNI_SIMD_BACKEND_AARCH64_NEON, "NEON report");
        }
        if (std::find(resolved.begin(), resolved.end(), actual_backend) == resolved.end()) {
            resolved.push_back(actual_backend);
            const auto actual = Analyze(analyzer, unaligned, count);
            Compare(actual, expected, 8.0e-5f);
        }
        Check(uni_simd_kernel_free(analyzer) == UNI_SIMD_RESULT_SUCCESS, "free SIMD");
    }

    uni_simd_kernel_t* sse = nullptr;
    Check(TryCreate(UNI_SIMD_BACKEND_X86_SSE2, &sse) == UNI_SIMD_RESULT_UNSUPPORTED_BACKEND, "SSE2 unsupported for analyzer");
    Check(sse == nullptr, "SSE2 null");

    auto* const automatic = Create(UNI_SIMD_BACKEND_AUTOMATIC);
    const auto automatic_backend = Backend(automatic);
    Check(automatic_backend == UNI_SIMD_BACKEND_GENERIC || automatic_backend == UNI_SIMD_BACKEND_X86_AVX2_FMA ||
              automatic_backend == UNI_SIMD_BACKEND_X86_AVX512 || automatic_backend == UNI_SIMD_BACKEND_AARCH64_NEON,
          "automatic backend report");
    Compare(Analyze(automatic, unaligned, count), expected, 8.0e-5f);

    auto* const deterministic = Create(UNI_SIMD_BACKEND_GENERIC, UNI_SIMD_MATH_DETERMINISTIC);
    Check(Backend(deterministic) == UNI_SIMD_BACKEND_GENERIC, "deterministic backend");

    Check(uni_simd_kernel_free(generic) == UNI_SIMD_RESULT_SUCCESS, "free generic");
    Check(uni_simd_kernel_free(automatic) == UNI_SIMD_RESULT_SUCCESS, "free auto");
    Check(uni_simd_kernel_free(deterministic) == UNI_SIMD_RESULT_SUCCESS, "free deterministic");
}

void TestValidationAndSingleSample() {
    uni_simd_qpsk_carrier_analyzer_config_t config{};
    config.descriptor_size = UNI_SIMD_QPSK_CARRIER_ANALYZER_CONFIG_DESCRIPTOR_SIZE;
    config.magnitude_epsilon = 0.0f;
    auto* analyzer = uni_simd_kernel_create(UNI_SIMD_KERNEL_QPSK_CARRIER_ANALYZER_CF32);
    Check(analyzer != nullptr, "validation create");
    Check(uni_simd_kernel_param_set(analyzer, {UNI_SIMD_PARAM_CONFIG, {.const_pointer = &config}}) == UNI_SIMD_RESULT_SUCCESS,
          "zero epsilon configure");
    const uni_simd_qpsk_carrier_analyzer_block_t empty{UNI_SIMD_QPSK_CARRIER_ANALYZER_BLOCK_DESCRIPTOR_SIZE, nullptr, 0U};
    uni_simd_qpsk_carrier_analyzer_result_t invalid_result{};
    Check(uni_simd_kernel_execute(analyzer, &empty, &invalid_result) == UNI_SIMD_RESULT_INVALID_ARGUMENT, "zero epsilon invalid");
    config.magnitude_epsilon = std::numeric_limits<float>::infinity();
    Check(uni_simd_kernel_execute(analyzer, &empty, &invalid_result) == UNI_SIMD_RESULT_INVALID_ARGUMENT, "infinite epsilon invalid");
    Check(uni_simd_kernel_free(analyzer) == UNI_SIMD_RESULT_SUCCESS, "free invalid");

    analyzer = Create(UNI_SIMD_BACKEND_GENERIC);
    const std::array<float, 2U> one{1.0f, -1.0f};
    const auto first = Analyze(analyzer, one.data(), 1U);
    Check(first.valid_fourth_count == 1U && first.adjacent_fourth_count == 0U, "one sample");
    const auto second = Analyze(analyzer, one.data(), 1U);
    Check(second.adjacent_fourth_count == 1U, "one-sample boundary");
    Check(uni_simd_kernel_reset(nullptr) == UNI_SIMD_RESULT_INVALID_ARGUMENT, "null reset");
    Check(uni_simd_kernel_free(nullptr) == UNI_SIMD_RESULT_SUCCESS, "null free");
    Check(uni_simd_kernel_free(analyzer) == UNI_SIMD_RESULT_SUCCESS, "free validation");
}

} // namespace

int main() {
    Check(uni_simd_initialize() == UNI_SIMD_RESULT_SUCCESS, "initialize");
    TestAnalyticSigns();
    TestStreamingResetAndZeros();
    TestBackendsOddAndUnaligned();
    TestValidationAndSingleSample();
    Check(uni_simd_finalize() == UNI_SIMD_RESULT_SUCCESS, "finalize");
    return 0;
}
