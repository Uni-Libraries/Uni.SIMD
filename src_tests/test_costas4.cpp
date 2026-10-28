#include "uni_simd.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

void Check(const bool condition, const char* const message = "check") {
    if (!condition) {
        std::fprintf(stderr, "costas4 test failed: %s\n", message);
        std::abort();
    }
}

using Channel = std::vector<std::complex<float>>;

struct OracleState {
    float phase{};
    float phase_cos{1.0f};
    float phase_sin{};
    float frequency{};
    // Steps decided but not yet applied, oldest first.
    std::array<float, UNI_SIMD_QPSK_COSTAS4_LOOP_DELAY> pending_step{};
    std::array<float, UNI_SIMD_QPSK_COSTAS4_LOOP_DELAY> pending_step_cos = [] {
        std::array<float, UNI_SIMD_QPSK_COSTAS4_LOOP_DELAY> ones{};
        ones.fill(1.0f);
        return ones;
    }();
    std::array<float, UNI_SIMD_QPSK_COSTAS4_LOOP_DELAY> pending_step_sin{};
    std::size_t samples_since_normalization{};
};

constexpr float ProductionAlpha() {
    constexpr double bandwidth = 0.0001;
    constexpr double damping = 0.707;
    return static_cast<float>((4.0 * damping * bandwidth) / (1.0 + 2.0 * damping * bandwidth + bandwidth * bandwidth));
}

constexpr float ProductionBeta() {
    constexpr double bandwidth = 0.0001;
    constexpr double damping = 0.707;
    return static_cast<float>((4.0 * bandwidth * bandwidth) / (1.0 + 2.0 * damping * bandwidth + bandwidth * bandwidth));
}

void ProcessOracle(Channel& channel, const std::size_t offset, const std::size_t count, const float gain, OracleState& state) {
    const auto normalize = [&]() {
        constexpr float pi = 3.14159265358979323846f;
        constexpr float two_pi = 2.0f * pi;
        if (state.phase > pi) state.phase -= two_pi;
        else if (state.phase <= -pi) state.phase += two_pi;
        const float norm2 = state.phase_cos * state.phase_cos + state.phase_sin * state.phase_sin;
        const float inverse = 1.0f / std::sqrt(norm2);
        state.phase_cos *= inverse;
        state.phase_sin *= inverse;
    };
    for (std::size_t index = 0U; index < count; ++index) {
        auto& sample = channel[offset + index];
        const float real = sample.real() * gain;
        const float imag = sample.imag() * gain;
        const float output_real = std::fma(real, state.phase_cos, imag * state.phase_sin);
        const float output_imag = std::fma(imag, state.phase_cos, -(real * state.phase_sin));
        sample = {output_real, output_imag};
        const float decision_real = std::copysign(1.0f, output_real);
        const float decision_imag = std::copysign(1.0f, output_imag);
        const float error = decision_real * output_imag - decision_imag * output_real;
        state.frequency = std::clamp(std::fma(ProductionBeta(), error, state.frequency), -0.1f, 0.1f);
        const float delta = std::fma(ProductionAlpha(), error, state.frequency);
        const float squared = delta * delta;
        const float delta_sin = std::fma(delta * squared, std::fma(squared, 1.0f / 120.0f, -1.0f / 6.0f), delta);
        const float delta_cos = std::fma(squared, std::fma(squared, 1.0f / 24.0f, -0.5f), 1.0f);
        // Loop delay: apply the step decided UNI_SIMD_QPSK_COSTAS4_LOOP_DELAY samples ago.
        const float previous_cos = state.phase_cos;
        const float previous_sin = state.phase_sin;
        state.phase_cos = std::fma(previous_cos, state.pending_step_cos[0], -(previous_sin * state.pending_step_sin[0]));
        state.phase_sin = std::fma(previous_sin, state.pending_step_cos[0], previous_cos * state.pending_step_sin[0]);
        state.phase += state.pending_step[0];
        std::shift_left(state.pending_step.begin(), state.pending_step.end(), 1);
        std::shift_left(state.pending_step_cos.begin(), state.pending_step_cos.end(), 1);
        std::shift_left(state.pending_step_sin.begin(), state.pending_step_sin.end(), 1);
        state.pending_step.back() = delta;
        state.pending_step_cos.back() = delta_cos;
        state.pending_step_sin.back() = delta_sin;
        if (++state.samples_since_normalization == 512U) {
            state.samples_since_normalization = 0U;
            normalize();
        }
    }
}

[[nodiscard]] std::array<Channel, 4U> Input() {
    std::array<Channel, 4U> result;
    for (std::size_t lane = 0U; lane < result.size(); ++lane) {
        result[lane].resize(262145U);
        float phase = 0.17f * static_cast<float>(lane + 1U);
        for (std::size_t index = 0U; index < result[lane].size(); ++index) {
            const float symbol_real = ((index / 3U + lane) & 1U) != 0U ? 1.0f : -1.0f;
            const float symbol_imag = ((index / 7U + lane) & 1U) != 0U ? 1.0f : -1.0f;
            result[lane][index] = std::complex<float>{symbol_real, symbol_imag} * std::polar(0.005f, phase);
            phase += 0.003f * static_cast<float>(lane + 1U);
        }
    }
    return result;
}

[[nodiscard]] uni_simd_kernel_t* Create(const uni_simd_backend_e backend) {
    uni_simd_qpsk_costas4_config_t config{};
    config.descriptor_size = UNI_SIMD_QPSK_COSTAS4_CONFIG_DESCRIPTOR_SIZE;
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
        config.alpha[lane] = ProductionAlpha();
        config.beta[lane] = ProductionBeta();
        config.error_clip[lane] = 0.0f;
    }
    auto* const result = uni_simd_kernel_create(UNI_SIMD_KERNEL_QPSK_COSTAS4_CF32);
    Check(result != nullptr);
    std::array params{
        uni_simd_param_t{UNI_SIMD_PARAM_CONFIG, {.const_pointer = &config}},
        uni_simd_param_t{UNI_SIMD_PARAM_BACKEND, {.u32 = backend}},
    };
    Check(uni_simd_kernel_param_set_many(result, params.data(), params.size()) == UNI_SIMD_RESULT_SUCCESS);
    uni_simd_qpsk_costas4_block_t empty{};
    empty.descriptor_size = UNI_SIMD_QPSK_COSTAS4_BLOCK_DESCRIPTOR_SIZE;
    for (float& gain : empty.input_gain) {
        gain = 1.0f;
    }
    Check(uni_simd_kernel_execute(result, &empty, nullptr) == UNI_SIMD_RESULT_SUCCESS);
    return result;
}

void Process(uni_simd_kernel_t* kernel, std::array<Channel, 4U>& channels, const std::size_t offset, const std::size_t count) {
    uni_simd_qpsk_costas4_block_t block{};
    block.descriptor_size = UNI_SIMD_QPSK_COSTAS4_BLOCK_DESCRIPTOR_SIZE;
    block.sample_count = count;
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
        block.channels[lane] = reinterpret_cast<float*>(channels[lane].data() + offset);
        block.input_gain[lane] = 80.0f + 40.0f * static_cast<float>(lane);
        block.frequency_limit[lane] = 0.1f;
    }
    Check(uni_simd_kernel_execute(kernel, &block, nullptr) == UNI_SIMD_RESULT_SUCCESS);
}

void TestCopyStateSkipAndRotate() {
    const auto input = Input();
    constexpr std::size_t kSplit = 100000U;
    const std::size_t total = input[0].size();

    // The reference runs through the whole stream.
    auto reference = input;
    auto* const serial = Create(UNI_SIMD_BACKEND_AUTOMATIC);
    Process(serial, reference, 0U, kSplit);
    auto* const copy = Create(UNI_SIMD_BACKEND_AUTOMATIC);
    Check(uni_simd_kernel_copy_state(copy, serial) == UNI_SIMD_RESULT_SUCCESS, "copy state");
    auto* const rotated = Create(UNI_SIMD_BACKEND_AUTOMATIC);
    Check(uni_simd_kernel_copy_state(rotated, serial) == UNI_SIMD_RESULT_SUCCESS, "copy state");
    auto* const skipped = Create(UNI_SIMD_BACKEND_AUTOMATIC);
    Check(uni_simd_kernel_copy_state(skipped, serial) == UNI_SIMD_RESULT_SUCCESS, "copy state");
    auto* const idle = Create(UNI_SIMD_BACKEND_AUTOMATIC);
    Check(uni_simd_kernel_copy_state(idle, serial) == UNI_SIMD_RESULT_SUCCESS, "copy state");
    Process(serial, reference, kSplit, total - kSplit);

    // A copy continues bit for bit.
    auto copied = input;
    Process(copy, copied, kSplit, total - kSplit);
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
        Check(std::equal(copied[lane].begin() + kSplit, copied[lane].end(), reference[lane].begin() + kSplit), "a copied state continues identically");
    }

    // Skipping no samples changes nothing.
    Check(uni_simd_qpsk_costas4_skip(idle, 0U) == UNI_SIMD_RESULT_SUCCESS, "skip nothing");
    auto idled = input;
    Process(idle, idled, kSplit, total - kSplit);
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
        Check(std::equal(idled[lane].begin() + kSplit, idled[lane].end(), reference[lane].begin() + kSplit), "skipping no samples is a no-op");
    }

    // A quarter-turn rotation multiplies the output by (-i)^turns and leaves the loop as it was:
    // the QPSK phase detector does not tell the four lock points apart.
    const std::array<std::int32_t, 4U> turns{0, 1, 2, -1};
    Check(uni_simd_qpsk_costas4_rotate(rotated, turns.data()) == UNI_SIMD_RESULT_SUCCESS, "rotate");
    auto turned = input;
    Process(rotated, turned, kSplit, total - kSplit);
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
        std::complex<float> factor{1.0f, 0.0f};
        for (std::int32_t turn = 0; turn < (turns[lane] & 3); ++turn) {
            factor *= std::complex<float>{0.0f, -1.0f};
        }
        for (std::size_t sample = kSplit; sample < total; sample += 97U) {
            Check(std::abs(turned[lane][sample] - reference[lane][sample] * factor) < 1e-3f, "rotation turns the output");
        }
    }

    // Coasting over a gap, the loops settle back onto the reference's trajectory (up to a lock
    // point). Needs a signal the loop locks on: QPSK with a carrier offset well inside its range.
    std::array<Channel, 4U> carrier;
    std::uint32_t random = 7U;
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
        carrier[lane].resize(total);
        std::complex<float> symbol{};
        for (std::size_t sample = 0U; sample < total; ++sample) {
            if (sample % 4U == 0U) {
                random = random * 1664525U + 1013904223U;
                symbol = {(random >> 30U & 1U) != 0U ? 1.0f : -1.0f, (random >> 29U & 1U) != 0U ? 1.0f : -1.0f};
            }
            const double phase = 0.4 * static_cast<double>(lane) + 2e-5 * static_cast<double>(lane + 1U) * static_cast<double>(sample);
            carrier[lane][sample] = symbol * std::polar(0.005f, static_cast<float>(std::remainder(phase, 2.0 * std::acos(-1.0))));
        }
    }
    auto locked = carrier;
    auto* const locked_serial = Create(UNI_SIMD_BACKEND_AUTOMATIC);
    Process(locked_serial, locked, 0U, kSplit);
    Check(uni_simd_kernel_copy_state(skipped, locked_serial) == UNI_SIMD_RESULT_SUCCESS, "copy state");
    Process(locked_serial, locked, kSplit, total - kSplit);
    constexpr std::size_t kGap = 60000U;
    Check(uni_simd_qpsk_costas4_skip(skipped, kGap) == UNI_SIMD_RESULT_SUCCESS, "skip");
    auto resumed = carrier;
    Process(skipped, resumed, kSplit + kGap, total - kSplit - kGap);
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
        std::complex<double> correlation{};
        float spread = 0.0f;
        for (std::size_t sample = total - 20000U; sample < total; ++sample) {
            correlation += std::complex<double>(resumed[lane][sample]) * std::conj(std::complex<double>(locked[lane][sample]));
            // Locked: the output lies on the axes-diagonal QPSK points, |re| == |im|.
            spread = std::max(spread, std::abs(std::abs(locked[lane][sample].real()) - std::abs(locked[lane][sample].imag())) / std::abs(locked[lane][sample]));
        }
        Check(spread < 0.05f, "the reference loop is locked");
        const double quarter = std::round(std::arg(correlation) / (std::acos(-1.0) / 2.0));
        const auto factor = std::polar(1.0f, static_cast<float>(-quarter * std::acos(-1.0) / 2.0));
        float worst = 0.0f;
        for (std::size_t sample = total - 20000U; sample < total; ++sample) {
            worst = std::max(worst, std::abs(resumed[lane][sample] * factor - locked[lane][sample]) / std::abs(locked[lane][sample]));
        }
        Check(worst < 1e-3f, "the coasted loop settles onto the reference");
    }
    Check(uni_simd_kernel_free(locked_serial) == UNI_SIMD_RESULT_SUCCESS);
    for (auto* const kernel : {serial, copy, rotated, skipped, idle}) {
        Check(uni_simd_kernel_free(kernel) == UNI_SIMD_RESULT_SUCCESS);
    }
}

} // namespace

int main() {
    Check(uni_simd_initialize() == UNI_SIMD_RESULT_SUCCESS);
    auto generic_input = Input();
    auto automatic_input = generic_input;
    auto oracle_input = generic_input;
    std::array<OracleState, 4U> oracle_state{};
    auto* generic = Create(UNI_SIMD_BACKEND_GENERIC);
    auto* automatic = Create(UNI_SIMD_BACKEND_AUTOMATIC);
    Process(generic, generic_input, 0U, 131071U);
    Process(generic, generic_input, 131071U, generic_input[0].size() - 131071U);
    Process(automatic, automatic_input, 0U, 131071U);
    Process(automatic, automatic_input, 131071U, automatic_input[0].size() - 131071U);
    for (std::size_t lane = 0U; lane < 4U; ++lane) ProcessOracle(oracle_input[lane], 0U, 131071U, 80.0f + 40.0f * static_cast<float>(lane), oracle_state[lane]);
    for (std::size_t lane = 0U; lane < 4U; ++lane) ProcessOracle(oracle_input[lane], 131071U, oracle_input[lane].size() - 131071U,
                                                                 80.0f + 40.0f * static_cast<float>(lane), oracle_state[lane]);
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
        for (std::size_t sample = 0U; sample < generic_input[lane].size(); ++sample) {
            Check(generic_input[lane][sample] == automatic_input[lane][sample], "generic != automatic");
            Check(generic_input[lane][sample] == oracle_input[lane][sample], "generic != oracle");
        }
    }
    Check(uni_simd_kernel_reset(automatic) == UNI_SIMD_RESULT_SUCCESS);
    Check(uni_simd_kernel_free(generic) == UNI_SIMD_RESULT_SUCCESS);
    Check(uni_simd_kernel_free(automatic) == UNI_SIMD_RESULT_SUCCESS);

    uni_simd_qpsk_costas4_config_t huge_phase_config{};
    huge_phase_config.descriptor_size = UNI_SIMD_QPSK_COSTAS4_CONFIG_DESCRIPTOR_SIZE;
    huge_phase_config.initial_phase[0] = std::numeric_limits<float>::max();
    auto* const huge_phase = uni_simd_kernel_create(UNI_SIMD_KERNEL_QPSK_COSTAS4_CF32);
    Check(huge_phase != nullptr);
    Check(uni_simd_kernel_param_set(huge_phase, {UNI_SIMD_PARAM_CONFIG, {.const_pointer = &huge_phase_config}}) == UNI_SIMD_RESULT_SUCCESS);
    uni_simd_qpsk_costas4_block_t empty{};
    empty.descriptor_size = UNI_SIMD_QPSK_COSTAS4_BLOCK_DESCRIPTOR_SIZE;
    for (float& gain : empty.input_gain) {
        gain = 1.0f;
    }
    uni_simd_qpsk_costas4_state_t huge_phase_state{};
    Check(uni_simd_kernel_execute(huge_phase, &empty, &huge_phase_state) == UNI_SIMD_RESULT_SUCCESS);
    Check(std::isfinite(huge_phase_state.phase[0]), "huge phase normalized");
    Check(uni_simd_kernel_free(huge_phase) == UNI_SIMD_RESULT_SUCCESS);
    TestCopyStateSkipAndRotate();
    Check(uni_simd_finalize() == UNI_SIMD_RESULT_SUCCESS);
    return 0;
}
