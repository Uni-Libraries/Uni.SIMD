#ifdef NDEBUG
#undef NDEBUG
#endif

#include <uni/simd/uni_simd.h>

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static uni_simd_param_t u32_param(uni_simd_param_id_e id, uint32_t value) {
    uni_simd_param_t param = {id, UNI_SIMD_PARAM_TYPE_U32, {0}};
    param.value.u32 = value;
    return param;
}

static uni_simd_param_t size_param(uni_simd_param_id_e id, size_t value) {
    uni_simd_param_t param = {id, UNI_SIMD_PARAM_TYPE_SIZE, {0}};
    param.value.size = value;
    return param;
}

static uni_simd_param_t f32_param(uni_simd_param_id_e id, float value) {
    uni_simd_param_t param = {id, UNI_SIMD_PARAM_TYPE_FLOAT32, {0}};
    param.value.f32 = value;
    return param;
}

static uni_simd_param_t pointer_param(uni_simd_param_id_e id, const void* value) {
    uni_simd_param_t param = {id, UNI_SIMD_PARAM_TYPE_CONST_POINTER, {0}};
    param.value.const_pointer = value;
    return param;
}

int main(void) {
    const uint8_t bits[16] = {1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1};
    uint8_t packed[2] = {0, 0};
    const uni_simd_const_buffer_t bits_input = {bits, 16U};
    uni_simd_buffer_t packed_output = {packed, 2U};

    assert(uni_simd_execute(UNI_SIMD_KERNEL_PACK_BITS_LSB_U8, &bits_input, &packed_output,
                            NULL, 0U, NULL) == UNI_SIMD_RESULT_NOT_INITIALIZED);
    assert(uni_simd_initialize() == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_initialize() == UNI_SIMD_RESULT_SUCCESS);

    uni_simd_param_t backend_params[] = {
        u32_param(UNI_SIMD_PARAM_BACKEND, UNI_SIMD_BACKEND_GENERIC),
        u32_param(UNI_SIMD_PARAM_RESOLVED_BACKEND, UNI_SIMD_BACKEND_AUTOMATIC),
    };
    assert(uni_simd_execute(UNI_SIMD_KERNEL_PACK_BITS_LSB_U8, &bits_input, &packed_output,
                            backend_params, 2U, NULL) == UNI_SIMD_RESULT_SUCCESS);
    assert(packed[0] == 0x4dU && packed[1] == 0x96U);
    assert(backend_params[1].value.u32 == UNI_SIMD_BACKEND_GENERIC);

    uint8_t unpacked[16] = {0};
    const uni_simd_const_buffer_t packed_input = {packed, 2U};
    uni_simd_buffer_t unpacked_output = {unpacked, 16U};
    assert(uni_simd_execute(UNI_SIMD_KERNEL_UNPACK_BITS_LSB_U8, &packed_input, &unpacked_output,
                            NULL, 0U, NULL) == UNI_SIMD_RESULT_SUCCESS);
    for (size_t index = 0U; index < 16U; ++index) {
        assert(unpacked[index] == bits[index]);
    }
    assert(uni_simd_execute(UNI_SIMD_KERNEL_PACK_BITS_MSB_U8, &bits_input, &packed_output,
                            NULL, 0U, NULL) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_execute(UNI_SIMD_KERNEL_UNPACK_BITS_MSB_U8, &packed_input, &unpacked_output,
                            NULL, 0U, NULL) == UNI_SIMD_RESULT_SUCCESS);
    for (size_t index = 0U; index < 16U; ++index) {
        assert(unpacked[index] == bits[index]);
    }

    uint8_t byte_input_data[3] = {0x00U, 0x55U, 0xffU};
    uint8_t byte_output_data[3] = {0U, 0U, 0U};
    const uni_simd_const_buffer_t byte_input = {byte_input_data, 3U};
    uni_simd_buffer_t byte_output = {byte_output_data, 3U};
    assert(uni_simd_execute(UNI_SIMD_KERNEL_COPY_U8, &byte_input, &byte_output,
                            NULL, 0U, NULL) == UNI_SIMD_RESULT_SUCCESS);
    assert(byte_output_data[0] == 0x00U && byte_output_data[1] == 0x55U && byte_output_data[2] == 0xffU);
    assert(uni_simd_execute(UNI_SIMD_KERNEL_INVERT_LSB_U8, &byte_input, &byte_output,
                            NULL, 0U, NULL) == UNI_SIMD_RESULT_SUCCESS);
    assert(byte_output_data[0] == 0x01U && byte_output_data[1] == 0x54U && byte_output_data[2] == 0xfeU);
    assert(uni_simd_execute(UNI_SIMD_KERNEL_INVERT_U8, &byte_input, &byte_output,
                            NULL, 0U, NULL) == UNI_SIMD_RESULT_SUCCESS);
    assert(byte_output_data[0] == 0xffU && byte_output_data[1] == 0xaaU && byte_output_data[2] == 0x00U);

    float symbols[4] = {1.0f, -1.0f, 0.5f, -0.5f};
    uint8_t quantized[4] = {0, 0, 0, 0};
    const uni_simd_const_buffer_t symbol_input = {symbols, 2U};
    uni_simd_buffer_t quantized_output = {quantized, 4U};
    uni_simd_param_t quantize_params[] = {
        f32_param(UNI_SIMD_PARAM_SCALE, -10.0f),
        f32_param(UNI_SIMD_PARAM_OFFSET, 128.0f),
    };
    assert(uni_simd_execute(UNI_SIMD_KERNEL_QUANTIZE_CF32_U8, &symbol_input, &quantized_output,
                            quantize_params, 2U, NULL) == UNI_SIMD_RESULT_SUCCESS);
    assert(quantized[0] == 118U && quantized[1] == 138U && quantized[2] == 123U && quantized[3] == 133U);

    float powers[2] = {0.0f, 0.0f};
    uni_simd_buffer_t power_output = {powers, 2U};
    assert(uni_simd_execute(UNI_SIMD_KERNEL_MAGNITUDE_SQUARED_CF32_F32, &symbol_input, &power_output,
                            NULL, 0U, NULL) == UNI_SIMD_RESULT_SUCCESS);
    assert(fabsf(powers[0] - 2.0f) < 1.0e-6f && fabsf(powers[1] - 0.5f) < 1.0e-6f);
    uni_simd_param_t psd_params[] = {f32_param(UNI_SIMD_PARAM_RBW_HZ, 2.0f)};
    assert(uni_simd_execute(UNI_SIMD_KERNEL_POWER_SPECTRAL_DENSITY_CF32_F32,
                            &symbol_input, &power_output, psd_params, 1U, NULL) == UNI_SIMD_RESULT_SUCCESS);
    assert(fabsf(powers[0] - 1.0f) < 1.0e-6f && fabsf(powers[1] - 0.25f) < 1.0e-6f);

    const float wide_symbol[2] = {1.0e20f, 0.0f};
    const uni_simd_const_buffer_t wide_input = {wide_symbol, 1U};
    float deterministic_power = 0.0f;
    uni_simd_buffer_t deterministic_output = {&deterministic_power, 1U};
    uni_simd_param_t deterministic_params[] = {
        u32_param(UNI_SIMD_PARAM_MATH_MODE, UNI_SIMD_MATH_DETERMINISTIC),
        f32_param(UNI_SIMD_PARAM_NORMALIZATION_FACTOR, 1.0e20f),
    };
    assert(uni_simd_execute(UNI_SIMD_KERNEL_MAGNITUDE_SQUARED_CF32_F32,
                            &wide_input, &deterministic_output, deterministic_params, 2U, NULL) ==
           UNI_SIMD_RESULT_SUCCESS);
    assert(isfinite(deterministic_power) && fabsf(deterministic_power - 1.0f) < 1.0e-6f);

    const float taps[2] = {0.5f, 2.0f};
    float dot[2] = {0.0f, 0.0f};
    uni_simd_buffer_t dot_output = {dot, 1U};
    uni_simd_param_t dot_params[] = {
        pointer_param(UNI_SIMD_PARAM_TAPS, taps),
        size_param(UNI_SIMD_PARAM_TAP_COUNT, 2U),
    };
    assert(uni_simd_execute(UNI_SIMD_KERNEL_DOT_CF32_F32, &symbol_input, &dot_output,
                            dot_params, 2U, NULL) == UNI_SIMD_RESULT_SUCCESS);
    assert(fabsf(dot[0] - 1.5f) < 1.0e-6f && fabsf(dot[1] + 1.5f) < 1.0e-6f);

    const float symmetric_symbols[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float symmetric_taps[1] = {0.5f};
    const uni_simd_const_buffer_t symmetric_input = {symmetric_symbols, 3U};
    uni_simd_param_t symmetric_params[] = {
        pointer_param(UNI_SIMD_PARAM_TAPS, symmetric_taps),
        size_param(UNI_SIMD_PARAM_TAP_COUNT, 1U),
        f32_param(UNI_SIMD_PARAM_CENTER_TAP, 2.0f),
    };
    assert(uni_simd_execute(UNI_SIMD_KERNEL_DOT_SYMMETRIC_CF32_F32, &symmetric_input, &dot_output,
                            symmetric_params, 3U, NULL) == UNI_SIMD_RESULT_SUCCESS);
    assert(fabsf(dot[0] - 9.0f) < 1.0e-6f && fabsf(dot[1] - 12.0f) < 1.0e-6f);

    float real[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float imag[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uni_simd_split_cf32_t split = {real, imag, 8U};
    assert(uni_simd_execute(UNI_SIMD_KERNEL_IFFT_SPLIT_CF32, NULL, &split,
                            backend_params, 2U, NULL) == UNI_SIMD_RESULT_SUCCESS);
    for (size_t index = 0U; index < 8U; ++index) {
        assert(fabsf(real[index] - 1.0f) < 1.0e-6f);
        assert(fabsf(imag[index]) < 1.0e-6f);
    }

    const float pfb_taps[1] = {1.0f};
    const int32_t logical_bins[1] = {0};
    uni_simd_param_t create_params[] = {
        u32_param(UNI_SIMD_PARAM_BACKEND, UNI_SIMD_BACKEND_GENERIC),
        size_param(UNI_SIMD_PARAM_BIN_COUNT, 8U),
        size_param(UNI_SIMD_PARAM_DECIMATION, 4U),
        u32_param(UNI_SIMD_PARAM_GRID_OFFSET, UNI_SIMD_PFB_INTEGER_BINS),
        pointer_param(UNI_SIMD_PARAM_TAPS, pfb_taps),
        size_param(UNI_SIMD_PARAM_TAP_COUNT, 1U),
        pointer_param(UNI_SIMD_PARAM_LOGICAL_BINS, logical_bins),
        size_param(UNI_SIMD_PARAM_LOGICAL_BIN_COUNT, 1U),
        u32_param(UNI_SIMD_PARAM_RESOLVED_BACKEND, UNI_SIMD_BACKEND_AUTOMATIC),
    };
    uni_simd_state_t* state = NULL;
    const float failed_input_samples[10] = {0.0f};
    float failed_output_samples[2] = {0.0f};
    const uni_simd_const_buffer_t failed_input = {failed_input_samples, 5U};
    uni_simd_buffer_t failed_output_buffer = {failed_output_samples, 1U};
    uni_simd_buffer_array_t failed_outputs = {&failed_output_buffer, 1U};
    assert(uni_simd_execute(UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32, &failed_input, &failed_outputs,
                            create_params, 9U, &state) == UNI_SIMD_RESULT_INVALID_SIZE);
    assert(state == NULL);
    assert(uni_simd_execute(UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32, NULL, NULL,
                            create_params, 9U, &state) == UNI_SIMD_RESULT_SUCCESS);
    assert(state != NULL && create_params[8].value.u32 == UNI_SIMD_BACKEND_GENERIC);
    assert(uni_simd_finalize() == UNI_SIMD_RESULT_INVALID_STATE);

    uni_simd_param_t query_params[] = {
        u32_param(UNI_SIMD_PARAM_QUERY_OUTPUT_COUNT, 1U),
        size_param(UNI_SIMD_PARAM_OUTPUT_COUNT, 0U),
    };
    const uni_simd_const_buffer_t query_input = {NULL, 5U};
    assert(uni_simd_execute(UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32, &query_input, NULL,
                            query_params, 2U, &state) == UNI_SIMD_RESULT_SUCCESS);
    assert(query_params[1].value.size == 2U);

    const float pfb_input_samples[10] = {
        1.0f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.25f,
    };
    float pfb_output_samples[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uni_simd_const_buffer_t pfb_input = {pfb_input_samples, 5U};
    uni_simd_buffer_t pfb_output_buffer = {pfb_output_samples, 2U};
    uni_simd_buffer_array_t pfb_outputs = {&pfb_output_buffer, 1U};
    uni_simd_param_t process_params[] = {size_param(UNI_SIMD_PARAM_OUTPUT_COUNT, 0U)};
    assert(uni_simd_execute(UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32, &pfb_input, &pfb_outputs,
                            process_params, 1U, &state) == UNI_SIMD_RESULT_SUCCESS);
    assert(process_params[0].value.size == 2U);
    assert(fabsf(pfb_output_samples[0] - 1.0f) < 1.0e-6f);
    assert(fabsf(pfb_output_samples[1] + 0.5f) < 1.0e-6f);
    assert(fabsf(pfb_output_samples[2] - 2.0f) < 1.0e-6f);
    assert(fabsf(pfb_output_samples[3] - 0.25f) < 1.0e-6f);

    uni_simd_param_t reset_params[] = {u32_param(UNI_SIMD_PARAM_RESET, 1U)};
    assert(uni_simd_execute(UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32, NULL, NULL,
                            reset_params, 1U, &state) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_execute(UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32, NULL, NULL,
                            NULL, 0U, &state) == UNI_SIMD_RESULT_INVALID_ARGUMENT);
    uni_simd_state_free(state);
    state = NULL;
    uni_simd_state_free(NULL);

    assert(uni_simd_finalize() == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_finalize() == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_execute(UNI_SIMD_KERNEL_COPY_U8, &bits_input, &packed_output,
                            NULL, 0U, NULL) == UNI_SIMD_RESULT_NOT_INITIALIZED);
    return 0;
}
