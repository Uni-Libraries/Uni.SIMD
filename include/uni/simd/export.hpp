#pragma once

#if defined(_WIN32)
#define UNI_SIMD_API
#elif defined(__GNUC__) || defined(__clang__)
#define UNI_SIMD_API __attribute__((visibility("default")))
#else
#define UNI_SIMD_API
#endif
