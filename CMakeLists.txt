cmake_minimum_required(VERSION 3.26)

project(UniSIMD VERSION 0.1.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include(CTest)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
endif()

option(UNI_SIMD_BUILD_TESTS "Build Uni.SIMD tests" ${PROJECT_IS_TOP_LEVEL})
option(UNI_SIMD_BUILD_BENCHMARKS "Build Uni.SIMD benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(UNI_SIMD_ENABLE_SANITIZERS "Enable ASan and UBSan for Uni.SIMD" OFF)

add_library(UniSIMD STATIC
    src/capabilities.cpp
    src/dispatch.cpp
    src/pfb_channelizer.cpp
    src/detail/pfb_channelizer_generic.cpp
    src/detail/simd_generic.cpp)
add_library(Uni::SIMD ALIAS UniSIMD)

target_compile_features(UniSIMD PUBLIC cxx_std_23)
target_include_directories(UniSIMD
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(UniSIMD PROPERTIES
    EXPORT_NAME SIMD
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

if(MSVC)
    set_property(SOURCE src/detail/pfb_channelizer_generic.cpp APPEND PROPERTY COMPILE_OPTIONS /fp:strict)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set_property(SOURCE src/detail/pfb_channelizer_generic.cpp APPEND PROPERTY COMPILE_OPTIONS -fno-fast-math -ffp-contract=off)
endif()

if(UNI_SIMD_ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(UniSIMD PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(UniSIMD INTERFACE -fsanitize=address,undefined)
    else()
        message(FATAL_ERROR "UNI_SIMD_ENABLE_SANITIZERS is supported with GCC and Clang")
    endif()
endif()

include(CheckCXXCompilerFlag)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" UNI_SIMD_SYSTEM_PROCESSOR)
set(UNI_SIMD_TARGET_X86 OFF)
set(UNI_SIMD_TARGET_AARCH64 OFF)
if(UNI_SIMD_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|x64|i[3-6]86)$" OR MSVC_CXX_ARCHITECTURE_ID MATCHES "^(X86|x64)$")
    set(UNI_SIMD_TARGET_X86 ON)
endif()
if(UNI_SIMD_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$" OR MSVC_CXX_ARCHITECTURE_ID MATCHES "^(ARM64)$")
    set(UNI_SIMD_TARGET_AARCH64 ON)
endif()
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    set(UNI_SIMD_TARGET_AARCH64 OFF)
    foreach(UNI_SIMD_OSX_ARCH IN LISTS CMAKE_OSX_ARCHITECTURES)
        if(NOT UNI_SIMD_OSX_ARCH MATCHES "^(x86_64|i386)$")
            set(UNI_SIMD_TARGET_X86 OFF)
        endif()
        if(UNI_SIMD_OSX_ARCH STREQUAL "arm64")
            set(UNI_SIMD_TARGET_AARCH64 ON)
        endif()
    endforeach()
endif()

if(UNI_SIMD_TARGET_X86)
    if(MSVC)
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(UNI_SIMD_HAVE_SSE2 ON)
            target_sources(UniSIMD PRIVATE src/detail/simd_sse2.cpp)
        else()
            check_cxx_compiler_flag("/arch:SSE2" UNI_SIMD_COMPILER_HAS_SSE2)
            if(UNI_SIMD_COMPILER_HAS_SSE2)
                set(UNI_SIMD_HAVE_SSE2 ON)
                target_sources(UniSIMD PRIVATE src/detail/simd_sse2.cpp)
                set_property(SOURCE src/detail/simd_sse2.cpp APPEND PROPERTY COMPILE_OPTIONS /arch:SSE2)
            endif()
        endif()

        check_cxx_compiler_flag("/arch:AVX2" UNI_SIMD_COMPILER_HAS_AVX2)
        if(UNI_SIMD_COMPILER_HAS_AVX2)
            set(UNI_SIMD_HAVE_AVX2 ON)
            set(UNI_SIMD_HAVE_AVX2_FMA ON)
            target_sources(UniSIMD PRIVATE src/detail/simd_avx2.cpp src/detail/simd_avx2fma.cpp
                                           src/detail/pfb_channelizer_avx2fma.cpp)
            set_property(SOURCE src/detail/simd_avx2.cpp src/detail/simd_avx2fma.cpp
                                src/detail/pfb_channelizer_avx2fma.cpp APPEND PROPERTY COMPILE_OPTIONS /arch:AVX2)
        endif()

        check_cxx_compiler_flag("/arch:AVX512" UNI_SIMD_COMPILER_HAS_AVX512)
        if(UNI_SIMD_COMPILER_HAS_AVX512)
            set(UNI_SIMD_HAVE_AVX512F ON)
            set(UNI_SIMD_HAVE_AVX512BW ON)
            target_sources(UniSIMD PRIVATE src/detail/simd_avx512f.cpp src/detail/simd_avx512bw.cpp)
            set_property(SOURCE src/detail/simd_avx512f.cpp src/detail/simd_avx512bw.cpp APPEND PROPERTY COMPILE_OPTIONS /arch:AVX512)
        endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        check_cxx_compiler_flag("-msse2" UNI_SIMD_COMPILER_HAS_SSE2)
        check_cxx_compiler_flag("-mavx2" UNI_SIMD_COMPILER_HAS_AVX2)
        check_cxx_compiler_flag("-mfma" UNI_SIMD_COMPILER_HAS_FMA)
        check_cxx_compiler_flag("-mavx512f" UNI_SIMD_COMPILER_HAS_AVX512F)
        check_cxx_compiler_flag("-mavx512bw" UNI_SIMD_COMPILER_HAS_AVX512BW)

        if(UNI_SIMD_COMPILER_HAS_SSE2)
            set(UNI_SIMD_HAVE_SSE2 ON)
            target_sources(UniSIMD PRIVATE src/detail/simd_sse2.cpp)
            set_property(SOURCE src/detail/simd_sse2.cpp APPEND PROPERTY COMPILE_OPTIONS -msse2)
        endif()
        if(UNI_SIMD_COMPILER_HAS_AVX2)
            set(UNI_SIMD_HAVE_AVX2 ON)
            target_sources(UniSIMD PRIVATE src/detail/simd_avx2.cpp)
            set_property(SOURCE src/detail/simd_avx2.cpp APPEND PROPERTY COMPILE_OPTIONS -mavx2)
        endif()
        if(UNI_SIMD_COMPILER_HAS_AVX2 AND UNI_SIMD_COMPILER_HAS_FMA)
            set(UNI_SIMD_HAVE_AVX2_FMA ON)
            target_sources(UniSIMD PRIVATE src/detail/simd_avx2fma.cpp src/detail/pfb_channelizer_avx2fma.cpp)
            set_property(SOURCE src/detail/simd_avx2fma.cpp src/detail/pfb_channelizer_avx2fma.cpp
                         APPEND PROPERTY COMPILE_OPTIONS -mavx2 -mfma)
        endif()
        if(UNI_SIMD_COMPILER_HAS_AVX512F)
            set(UNI_SIMD_HAVE_AVX512F ON)
            target_sources(UniSIMD PRIVATE src/detail/simd_avx512f.cpp)
            set_property(SOURCE src/detail/simd_avx512f.cpp APPEND PROPERTY COMPILE_OPTIONS -mavx512f)
        endif()
        if(UNI_SIMD_COMPILER_HAS_AVX512F AND UNI_SIMD_COMPILER_HAS_AVX512BW)
            set(UNI_SIMD_HAVE_AVX512BW ON)
            target_sources(UniSIMD PRIVATE src/detail/simd_avx512bw.cpp)
            set_property(SOURCE src/detail/simd_avx512bw.cpp APPEND PROPERTY COMPILE_OPTIONS -mavx512f -mavx512bw)
        endif()
    endif()
endif()

if(UNI_SIMD_TARGET_AARCH64)
    set(UNI_SIMD_HAVE_NEON ON)
    target_sources(UniSIMD PRIVATE src/detail/pfb_channelizer_neon.cpp)
endif()

foreach(UNI_SIMD_ISA SSE2 AVX2 AVX2_FMA AVX512F AVX512BW NEON)
    target_compile_definitions(UniSIMD PRIVATE UNI_SIMD_HAVE_${UNI_SIMD_ISA}=$<BOOL:${UNI_SIMD_HAVE_${UNI_SIMD_ISA}}>)
endforeach()

if(UNI_SIMD_BUILD_TESTS)
    add_executable(uni_simd_tests src_tests/test_simd.cpp)
    target_link_libraries(uni_simd_tests PRIVATE Uni::SIMD)
    target_compile_features(uni_simd_tests PRIVATE cxx_std_23)
    add_executable(uni_simd_pfb_channelizer_tests src_tests/test_pfb_channelizer.cpp)
    target_link_libraries(uni_simd_pfb_channelizer_tests PRIVATE Uni::SIMD)
    target_compile_features(uni_simd_pfb_channelizer_tests PRIVATE cxx_std_23)
    if(UNI_SIMD_ENABLE_SANITIZERS)
        foreach(UNI_SIMD_TEST_TARGET uni_simd_tests uni_simd_pfb_channelizer_tests)
            target_compile_options(${UNI_SIMD_TEST_TARGET} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${UNI_SIMD_TEST_TARGET} PRIVATE -fsanitize=address,undefined)
        endforeach()
    endif()
    add_test(NAME uni_simd_tests COMMAND uni_simd_tests)
    add_test(NAME uni_simd_pfb_channelizer_tests COMMAND uni_simd_pfb_channelizer_tests)
endif()

if(UNI_SIMD_BUILD_BENCHMARKS)
    add_subdirectory(src_benchmark)
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    configure_package_config_file(
        cmake/UniSIMDConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/UniSIMDConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/UniSIMD")
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/UniSIMDConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)
    install(TARGETS UniSIMD EXPORT UniSIMDTargets ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
    install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    install(EXPORT UniSIMDTargets FILE UniSIMDTargets.cmake NAMESPACE Uni:: DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/UniSIMD")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/UniSIMDConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/UniSIMDConfigVersion.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/UniSIMD")
endif()
