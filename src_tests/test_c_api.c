#include <uni_simd.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static uni_simd_param_val u32_val(uint32_t value) {
    uni_simd_param_val val = {0};
    val.u32 = value;
    return val;
}

static uni_simd_param_val size_val(size_t value) {
    uni_simd_param_val val = {0};
    val.size = value;
    return val;
}

static uni_simd_param_val f32_val(float value) {
    uni_simd_param_val val = {0};
    val.f32 = value;
    return val;
}

static uni_simd_param_val const_pointer_val(const void* value) {
    uni_simd_param_val val = {0};
    val.const_pointer = value;
    return val;
}

static uni_simd_param_val pointer_val(void* value) {
    uni_simd_param_val val = {0};
    val.pointer = value;
    return val;
}

static uni_simd_kernel_t* create_kernel(uni_simd_kernel_e id) {
    uni_simd_kernel_t* kernel = uni_simd_kernel_create(id);
    assert(kernel != NULL);
    return kernel;
}

static void free_kernel(uni_simd_kernel_t* kernel) {
    assert(uni_simd_kernel_free(kernel) == UNI_SIMD_RESULT_SUCCESS);
}

int main(void) {
    const uint8_t bits[16] = {1U, 0U, 1U, 1U, 0U, 0U, 1U, 0U, 0U, 1U, 1U, 0U, 1U, 0U, 0U, 1U};
    uint8_t packed[2] = {0U, 0U};
    const uni_simd_const_buffer_t bits_input = {bits, 16U};
    uni_simd_buffer_t packed_output = {packed, 2U};

    assert(uni_simd_kernel_create(UNI_SIMD_KERNEL_PACK_BITS_LSB_U8) == NULL);
    assert(uni_simd_initialize() == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_initialize() == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_create(UNI_SIMD_KERNEL_UNKNOWN) == NULL);
    assert(uni_simd_kernel_param_set(NULL, UNI_SIMD_PARAM_BACKEND, u32_val(0U)) ==
           UNI_SIMD_RESULT_INVALID_ARGUMENT);
    assert(uni_simd_kernel_execute(NULL, NULL, NULL) == UNI_SIMD_RESULT_INVALID_ARGUMENT);

    uni_simd_kernel_t* kernel = create_kernel(UNI_SIMD_KERNEL_PACK_BITS_LSB_U8);
    uni_simd_backend_e resolved_backend = UNI_SIMD_BACKEND_AUTOMATIC;
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_BACKEND,
                                     u32_val(UNI_SIMD_BACKEND_GENERIC)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_RESOLVED_BACKEND,
                                     pointer_val(&resolved_backend)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_SCALE, f32_val(1.0f)) ==
           UNI_SIMD_RESULT_INVALID_ARGUMENT);
    assert(uni_simd_kernel_execute(kernel, &bits_input, &packed_output) == UNI_SIMD_RESULT_SUCCESS);
    assert(resolved_backend == UNI_SIMD_BACKEND_GENERIC);
    assert(packed[0] == 0x4dU && packed[1] == 0x96U);
    free_kernel(kernel);

    uint8_t unpacked[16] = {0U};
    const uni_simd_const_buffer_t packed_input = {packed, 2U};
    uni_simd_buffer_t unpacked_output = {unpacked, 16U};
    kernel = create_kernel(UNI_SIMD_KERNEL_UNPACK_BITS_LSB_U8);
    assert(uni_simd_kernel_execute(kernel, &packed_input, &unpacked_output) == UNI_SIMD_RESULT_SUCCESS);
    for (size_t index = 0U; index < 16U; ++index) {
        assert(unpacked[index] == bits[index]);
    }
    free_kernel(kernel);

    kernel = create_kernel(UNI_SIMD_KERNEL_PACK_BITS_MSB_U8);
    assert(uni_simd_kernel_execute(kernel, &bits_input, &packed_output) == UNI_SIMD_RESULT_SUCCESS);
    free_kernel(kernel);
    kernel = create_kernel(UNI_SIMD_KERNEL_UNPACK_BITS_MSB_U8);
    assert(uni_simd_kernel_execute(kernel, &packed_input, &unpacked_output) == UNI_SIMD_RESULT_SUCCESS);
    for (size_t index = 0U; index < 16U; ++index) {
        assert(unpacked[index] == bits[index]);
    }
    free_kernel(kernel);

    const uint8_t byte_input_data[3] = {0x00U, 0x55U, 0xffU};
    uint8_t byte_output_data[3] = {0U, 0U, 0U};
    const uni_simd_const_buffer_t byte_input = {byte_input_data, 3U};
    uni_simd_buffer_t byte_output = {byte_output_data, 3U};
    kernel = create_kernel(UNI_SIMD_KERNEL_COPY_U8);
    assert(uni_simd_kernel_execute(kernel, &byte_input, &byte_output) == UNI_SIMD_RESULT_SUCCESS);
    assert(byte_output_data[0] == 0x00U && byte_output_data[1] == 0x55U && byte_output_data[2] == 0xffU);
    free_kernel(kernel);
    kernel = create_kernel(UNI_SIMD_KERNEL_INVERT_LSB_U8);
    assert(uni_simd_kernel_execute(kernel, &byte_input, &byte_output) == UNI_SIMD_RESULT_SUCCESS);
    assert(byte_output_data[0] == 0x01U && byte_output_data[1] == 0x54U && byte_output_data[2] == 0xfeU);
    free_kernel(kernel);
    kernel = create_kernel(UNI_SIMD_KERNEL_INVERT_U8);
    assert(uni_simd_kernel_execute(kernel, &byte_input, &byte_output) == UNI_SIMD_RESULT_SUCCESS);
    assert(byte_output_data[0] == 0xffU && byte_output_data[1] == 0xaaU && byte_output_data[2] == 0x00U);
    free_kernel(kernel);

    const float symbols[4] = {1.0f, -1.0f, 0.5f, -0.5f};
    uint8_t quantized[4] = {0U, 0U, 0U, 0U};
    const uni_simd_const_buffer_t symbol_input = {symbols, 2U};
    uni_simd_buffer_t quantized_output = {quantized, 4U};
    kernel = create_kernel(UNI_SIMD_KERNEL_QUANTIZE_CF32_U8);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_SCALE, f32_val(-10.0f)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_OFFSET, f32_val(128.0f)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_execute(kernel, &symbol_input, &quantized_output) == UNI_SIMD_RESULT_SUCCESS);
    assert(quantized[0] == 118U && quantized[1] == 138U && quantized[2] == 123U && quantized[3] == 133U);
    free_kernel(kernel);

    float powers[2] = {0.0f, 0.0f};
    uni_simd_buffer_t power_output = {powers, 2U};
    kernel = create_kernel(UNI_SIMD_KERNEL_MAGNITUDE_SQUARED_CF32_F32);
    assert(uni_simd_kernel_execute(kernel, &symbol_input, &power_output) == UNI_SIMD_RESULT_SUCCESS);
    assert(fabsf(powers[0] - 2.0f) < 1.0e-6f && fabsf(powers[1] - 0.5f) < 1.0e-6f);
    free_kernel(kernel);
    kernel = create_kernel(UNI_SIMD_KERNEL_POWER_SPECTRAL_DENSITY_CF32_F32);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_RBW_HZ, f32_val(2.0f)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_execute(kernel, &symbol_input, &power_output) == UNI_SIMD_RESULT_SUCCESS);
    assert(fabsf(powers[0] - 1.0f) < 1.0e-6f && fabsf(powers[1] - 0.25f) < 1.0e-6f);
    free_kernel(kernel);

    const float wide_symbol[2] = {1.0e20f, 0.0f};
    const uni_simd_const_buffer_t wide_input = {wide_symbol, 1U};
    float deterministic_power = 0.0f;
    uni_simd_buffer_t deterministic_output = {&deterministic_power, 1U};
    kernel = create_kernel(UNI_SIMD_KERNEL_MAGNITUDE_SQUARED_CF32_F32);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_MATH_MODE,
                                     u32_val(UNI_SIMD_MATH_DETERMINISTIC)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_NORMALIZATION_FACTOR,
                                     f32_val(1.0e20f)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_execute(kernel, &wide_input, &deterministic_output) == UNI_SIMD_RESULT_SUCCESS);
    assert(isfinite(deterministic_power) && fabsf(deterministic_power - 1.0f) < 1.0e-6f);
    free_kernel(kernel);

    const float taps[2] = {0.5f, 1.0f};
    float dot[2] = {0.0f, 0.0f};
    uni_simd_buffer_t dot_output = {dot, 1U};
    kernel = create_kernel(UNI_SIMD_KERNEL_DOT_CF32_F32);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_TAPS, const_pointer_val(taps)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_TAP_COUNT, size_val(2U)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_execute(kernel, &symbol_input, &dot_output) == UNI_SIMD_RESULT_SUCCESS);
    assert(fabsf(dot[0] - 1.0f) < 1.0e-6f && fabsf(dot[1] + 1.0f) < 1.0e-6f);
    free_kernel(kernel);

    const float symmetric_symbols[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float symmetric_taps[1] = {0.5f};
    const uni_simd_const_buffer_t symmetric_input = {symmetric_symbols, 3U};
    kernel = create_kernel(UNI_SIMD_KERNEL_DOT_SYMMETRIC_CF32_F32);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_TAPS,
                                     const_pointer_val(symmetric_taps)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_TAP_COUNT, size_val(1U)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_CENTER_TAP, f32_val(2.0f)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_execute(kernel, &symmetric_input, &dot_output) == UNI_SIMD_RESULT_SUCCESS);
    assert(fabsf(dot[0] - 9.0f) < 1.0e-6f && fabsf(dot[1] - 12.0f) < 1.0e-6f);
    free_kernel(kernel);

    float real[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float imag[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uni_simd_split_cf32_t split = {real, imag, UNI_SIMD_SPLIT_CF32_DESCRIPTOR_SIZE, 8U, 1U, 8U};
    kernel = create_kernel(UNI_SIMD_KERNEL_IFFT_SPLIT_CF32);
    assert(uni_simd_kernel_execute(kernel, NULL, &split) == UNI_SIMD_RESULT_SUCCESS);
    for (size_t index = 0U; index < 8U; ++index) {
        assert(fabsf(real[index] - 1.0f) < 1.0e-5f && fabsf(imag[index]) < 1.0e-5f);
    }
    float batch_real[18] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                            77.0f, 77.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float batch_imag[18] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                            -77.0f, -77.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uni_simd_split_cf32_t batch = {
        batch_real, batch_imag, UNI_SIMD_SPLIT_CF32_DESCRIPTOR_SIZE, 8U, 2U, 10U,
    };
    assert(uni_simd_kernel_execute(kernel, NULL, &batch) == UNI_SIMD_RESULT_SUCCESS);
    for (size_t index = 0U; index < 8U; ++index) {
        assert(fabsf(batch_real[index] - 1.0f) < 1.0e-6f && fabsf(batch_imag[index]) < 1.0e-6f);
        assert(fabsf(batch_real[10U + index]) < 1.0e-6f &&
               fabsf(batch_imag[10U + index] - 1.0f) < 1.0e-6f);
    }
    assert(batch_real[8] == 77.0f && batch_real[9] == 77.0f);
    assert(batch_imag[8] == -77.0f && batch_imag[9] == -77.0f);
    uni_simd_split_cf32_t empty_batch = {NULL, NULL, UNI_SIMD_SPLIT_CF32_DESCRIPTOR_SIZE, 8U, 0U, 0U};
    assert(uni_simd_kernel_execute(kernel, NULL, &empty_batch) == UNI_SIMD_RESULT_SUCCESS);
    split.transform_size = 3U;
    assert(uni_simd_kernel_execute(kernel, NULL, &split) == UNI_SIMD_RESULT_INVALID_SIZE);
    uni_simd_split_cf32_t invalid_stride = {
        batch_real, batch_imag, UNI_SIMD_SPLIT_CF32_DESCRIPTOR_SIZE, 8U, 2U, 7U,
    };
    assert(uni_simd_kernel_execute(kernel, NULL, &invalid_stride) == UNI_SIMD_RESULT_INVALID_SIZE);
    uni_simd_split_cf32_t legacy_descriptor = {real, imag, 8U, 0U, 0U, 0U};
    assert(uni_simd_kernel_execute(kernel, NULL, &legacy_descriptor) == UNI_SIMD_RESULT_INVALID_ARGUMENT);
    free_kernel(kernel);

    const float pfb_taps[1] = {1.0f};
    const int32_t logical_bins[1] = {0};
    size_t output_count = 0U;
    resolved_backend = UNI_SIMD_BACKEND_AUTOMATIC;
    kernel = create_kernel(UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_BACKEND,
                                     u32_val(UNI_SIMD_BACKEND_GENERIC)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_BIN_COUNT, size_val(8U)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_DECIMATION, size_val(4U)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_GRID_OFFSET,
                                     u32_val(UNI_SIMD_PFB_INTEGER_BINS)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_TAPS,
                                     const_pointer_val(pfb_taps)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_TAP_COUNT, size_val(1U)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_LOGICAL_BINS,
                                     const_pointer_val(logical_bins)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_LOGICAL_BIN_COUNT,
                                     size_val(1U)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_RESOLVED_BACKEND,
                                     pointer_val(&resolved_backend)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_OUTPUT_COUNT,
                                     pointer_val(&output_count)) == UNI_SIMD_RESULT_SUCCESS);

    const float failed_input_samples[10] = {0.0f};
    float failed_output_samples[2] = {0.0f};
    const uni_simd_const_buffer_t failed_input = {failed_input_samples, 5U};
    uni_simd_buffer_t failed_output_buffer = {failed_output_samples, 1U};
    uni_simd_buffer_array_t failed_outputs = {&failed_output_buffer, 1U};
    assert(uni_simd_kernel_execute(kernel, &failed_input, &failed_outputs) == UNI_SIMD_RESULT_INVALID_SIZE);
    assert(uni_simd_kernel_execute(kernel, NULL, NULL) == UNI_SIMD_RESULT_SUCCESS);
    assert(resolved_backend == UNI_SIMD_BACKEND_GENERIC && output_count == 0U);
    assert(uni_simd_finalize() == UNI_SIMD_RESULT_INVALID_STATE);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_BIN_COUNT, size_val(16U)) ==
           UNI_SIMD_RESULT_INVALID_STATE);

    const uni_simd_const_buffer_t query_input = {NULL, 5U};
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_QUERY_OUTPUT_COUNT, u32_val(1U)) ==
           UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_execute(kernel, &query_input, NULL) == UNI_SIMD_RESULT_SUCCESS);
    assert(output_count == 2U);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_RESET, u32_val(1U)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_execute(kernel, &query_input, NULL) == UNI_SIMD_RESULT_INVALID_ARGUMENT);

    const float pfb_input_samples[10] = {10.0f, 0.0f, 20.0f, 0.0f, 30.0f,
                                         0.0f, 40.0f, 0.0f, 50.0f, 0.0f};
    float pfb_output_samples[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uni_simd_const_buffer_t pfb_input = {pfb_input_samples, 5U};
    uni_simd_buffer_t pfb_output_buffer = {pfb_output_samples, 2U};
    uni_simd_buffer_array_t pfb_outputs = {&pfb_output_buffer, 1U};
    assert(uni_simd_kernel_execute(kernel, &pfb_input, &pfb_outputs) == UNI_SIMD_RESULT_SUCCESS);
    assert(output_count == 2U);
    assert(fabsf(pfb_output_samples[0] - 10.0f) < 1.0e-6f && fabsf(pfb_output_samples[1]) < 1.0e-6f);
    assert(fabsf(pfb_output_samples[2] - 50.0f) < 1.0e-6f && fabsf(pfb_output_samples[3]) < 1.0e-6f);

    const float failed_reset_sample[2] = {99.0f, 0.0f};
    const uni_simd_const_buffer_t failed_reset_input = {failed_reset_sample, 1U};
    uni_simd_buffer_t zero_capacity = {NULL, 0U};
    uni_simd_buffer_array_t zero_capacity_outputs = {&zero_capacity, 1U};
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_RESET, u32_val(1U)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_execute(kernel, &failed_reset_input, &zero_capacity_outputs) ==
           UNI_SIMD_RESULT_INVALID_SIZE);
    const float continuation_samples[8] = {10.0f, 0.0f, 20.0f, 0.0f, 30.0f, 0.0f, 40.0f, 0.0f};
    float continuation_output[2] = {0.0f, 0.0f};
    const uni_simd_const_buffer_t continuation_input = {continuation_samples, 4U};
    uni_simd_buffer_t continuation_buffer = {continuation_output, 1U};
    uni_simd_buffer_array_t continuation_outputs = {&continuation_buffer, 1U};
    assert(uni_simd_kernel_execute(kernel, &continuation_input, &continuation_outputs) ==
           UNI_SIMD_RESULT_SUCCESS);
    assert(fabsf(continuation_output[0] - 40.0f) < 1.0e-6f && fabsf(continuation_output[1]) < 1.0e-6f);

    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_RESET, u32_val(1U)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_execute(kernel, NULL, NULL) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_execute(kernel, NULL, NULL) == UNI_SIMD_RESULT_INVALID_ARGUMENT);
    free_kernel(kernel);
    assert(uni_simd_kernel_free(NULL) == UNI_SIMD_RESULT_SUCCESS);

    kernel = create_kernel(UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_BACKEND,
                                     u32_val(UNI_SIMD_BACKEND_GENERIC)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_BIN_COUNT, size_val(8U)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_DECIMATION, size_val(4U)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_GRID_OFFSET,
                                     u32_val(UNI_SIMD_PFB_INTEGER_BINS)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_TAPS,
                                     const_pointer_val(pfb_taps)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_TAP_COUNT, size_val(1U)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_LOGICAL_BINS,
                                     const_pointer_val(NULL)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_param_set(kernel, UNI_SIMD_PARAM_LOGICAL_BIN_COUNT,
                                     size_val(0U)) == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_execute(kernel, NULL, NULL) == UNI_SIMD_RESULT_SUCCESS);
    const uni_simd_const_buffer_t overflowing_input = {
        pfb_input_samples, SIZE_MAX / (2U * sizeof(float)) + 1U,
    };
    assert(uni_simd_kernel_execute(kernel, &overflowing_input, NULL) == UNI_SIMD_RESULT_INVALID_SIZE);
    free_kernel(kernel);

    assert(uni_simd_finalize() == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_finalize() == UNI_SIMD_RESULT_SUCCESS);
    assert(uni_simd_kernel_execute(NULL, &bits_input, &packed_output) == UNI_SIMD_RESULT_NOT_INITIALIZED);
    return 0;
}
