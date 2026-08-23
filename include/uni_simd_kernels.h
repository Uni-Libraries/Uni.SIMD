#pragma once

#include <stdint.h>

/**
 * Operations accepted by uni_simd_kernel_create().
 *
 * Unless stated otherwise, input points to uni_simd_const_buffer_t, output
 * points to uni_simd_buffer_t, buffers require only natural element alignment
 * (no SIMD alignment), and count is an element count rather than a byte count.
 * Empty buffers are valid no-ops.
 */
typedef uint32_t uni_simd_kernel_e;
enum {
    UNI_SIMD_KERNEL_UNKNOWN = 0,

    /** Copies bytes with memmove semantics. No parameters are required. */
    UNI_SIMD_KERNEL_COPY_U8 = 1,

    /** Flips bit zero in every byte. Exact in-place operation is supported. */
    UNI_SIMD_KERNEL_INVERT_LSB_U8 = 2,

    /** Flips every bit in every byte. Exact in-place operation is supported. */
    UNI_SIMD_KERNEL_INVERT_U8 = 3,

    /** Packs groups of eight byte-valued bits into bytes, first input in bit 0. */
    UNI_SIMD_KERNEL_PACK_BITS_LSB_U8 = 4,

    /** Packs groups of eight byte-valued bits into bytes, first input in bit 7. */
    UNI_SIMD_KERNEL_PACK_BITS_MSB_U8 = 5,

    /** Unpacks bytes to byte-valued bits, bit 0 first. */
    UNI_SIMD_KERNEL_UNPACK_BITS_LSB_U8 = 6,

    /** Unpacks bytes to byte-valued bits, bit 7 first. */
    UNI_SIMD_KERNEL_UNPACK_BITS_MSB_U8 = 7,

    /**
     * Maps interleaved {real, imaginary} float input to two interleaved uint8_t
     * values per sample. Optional finite FLOAT32 parameters: SCALE (default 1)
     * and OFFSET (default 128).
     */
    UNI_SIMD_KERNEL_QUANTIZE_CF32_U8 = 8,

    /**
     * Computes squared magnitude from interleaved complex floats to float.
     * Optional FLOAT32 NORMALIZATION_FACTOR defaults to 1 and must be finite
     * and positive.
     */
    UNI_SIMD_KERNEL_MAGNITUDE_SQUARED_CF32_F32 = 9,

    /**
     * Computes power spectral density from interleaved complex floats to float.
     * Finite positive FLOAT32 RBW_HZ is required. Optional finite positive
     * NORMALIZATION_FACTOR defaults to 1.
     */
    UNI_SIMD_KERNEL_POWER_SPECTRAL_DENSITY_CF32_F32 = 10,

    /**
     * Computes a complex dot product. Input data is an interleaved float array;
     * output data has capacity for two floats. CONST_POINTER TAPS and SIZE
     * TAP_COUNT are required and TAP_COUNT must equal the input sample count.
     */
    UNI_SIMD_KERNEL_DOT_CF32_F32 = 11,

    /**
     * Computes a symmetric complex/real dot product. CONST_POINTER TAPS, SIZE
     * TAP_COUNT, and FLOAT32 CENTER_TAP are required. Input count must be
     * 2 * TAP_COUNT + 1; output data has capacity for two floats.
     */
    UNI_SIMD_KERNEL_DOT_SYMMETRIC_CF32_F32 = 12,

    /**
     * Executes one or more unscaled positive-exponent IFFTs in place. Input is
     * NULL and output points to uni_simd_split_cf32_t. Supported transform
     * sizes: 4, 8, 16, 32. Batches are dispatched once and may be strided.
     */
    UNI_SIMD_KERNEL_IFFT_SPLIT_CF32 = 13,

    /**
     * Stateful streaming PFB channelizer. See uni_simd_kernel_execute() for its
     * creation, query, reset, and processing protocol.
     */
    UNI_SIMD_KERNEL_PFB_CHANNELIZER_CF32 = 14
};
