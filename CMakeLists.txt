cmake_minimum_required(VERSION 3.26)

project(UniSIMD VERSION 0.4.0 LANGUAGES C CXX)

set(UNI_SIMD_ABI_VERSION 4)

include(GNUInstallDirs)

if(PROJECT_IS_TOP_LEVEL)
    include(CTest)
    include(CMakePackageConfigHelpers)
endif()

option(UNI_SIMD_BUILD_TESTS "Build Uni.SIMD tests" ${PROJECT_IS_TOP_LEVEL})
option(UNI_SIMD_BUILD_BENCHMARKS "Build Uni.SIMD benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(UNI_SIMD_ENABLE_SANITIZERS "Enable ASan and UBSan for Uni.SIMD" OFF)

add_library(UniSIMD)
add_library(Uni::SIMD ALIAS UniSIMD)

include(GenerateExportHeader)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/")
generate_export_header(UniSIMD
    EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/generated/uni_simd_export.h"
    EXPORT_MACRO_NAME UNI_SIMD_API
    NO_EXPORT_MACRO_NAME UNI_SIMD_NO_EXPORT
    STATIC_DEFINE UNI_SIMD_STATIC)
configure_file(
    cmake/uni_simd_version.h.in
    "${CMAKE_CURRENT_BINARY_DIR}/generated/uni_simd_version.h"
    @ONLY)

target_compile_features(UniSIMD PRIVATE cxx_std_23)
target_include_directories(UniSIMD
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(UniSIMD
    PUBLIC $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:UNI_SIMD_STATIC>)
set(UNI_SIMD_LIBRARY_VERSION ${PROJECT_VERSION})
if(APPLE)
    set(UNI_SIMD_LIBRARY_VERSION "${UNI_SIMD_ABI_VERSION}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
endif()
set_target_properties(UniSIMD PROPERTIES
    EXPORT_NAME SIMD
    VERSION ${UNI_SIMD_LIBRARY_VERSION}
    SOVERSION ${UNI_SIMD_ABI_VERSION}
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
if(WIN32)
    set_target_properties(UniSIMD PROPERTIES OUTPUT_NAME "UniSIMD-${UNI_SIMD_ABI_VERSION}")
endif()

include(cmake/UniSIMDIsaProfiles.cmake)
uni_simd_configure_isa_sources(UniSIMD "${CMAKE_CURRENT_SOURCE_DIR}/src")

set(UNI_SIMD_INTERNAL_TARGET UniSIMD)
if(BUILD_SHARED_LIBS AND UNI_SIMD_BUILD_TESTS)
    add_library(UniSIMDInternal STATIC)
    target_compile_features(UniSIMDInternal PRIVATE cxx_std_23)
    target_include_directories(UniSIMDInternal PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_definitions(UniSIMDInternal PRIVATE UNI_SIMD_STATIC)
    set_target_properties(UniSIMDInternal PROPERTIES
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
    uni_simd_configure_isa_sources(UniSIMDInternal "${CMAKE_CURRENT_SOURCE_DIR}/src")
    set(UNI_SIMD_INTERNAL_TARGET UniSIMDInternal)
endif()

if(UNI_SIMD_ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(UniSIMD PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(UniSIMD INTERFACE -fsanitize=address,undefined)
        if(TARGET UniSIMDInternal)
            target_compile_options(UniSIMDInternal PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        endif()
    else()
        message(FATAL_ERROR "UNI_SIMD_ENABLE_SANITIZERS is supported with GCC and Clang")
    endif()
endif()

if(UNI_SIMD_BUILD_TESTS)
    add_executable(uni_simd_tests src_tests/test_simd.cpp)
    target_link_libraries(uni_simd_tests PRIVATE ${UNI_SIMD_INTERNAL_TARGET})
    target_include_directories(uni_simd_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_features(uni_simd_tests PRIVATE cxx_std_23)
    add_executable(uni_simd_pfb_channelizer_tests src_tests/test_pfb_channelizer.cpp)
    target_link_libraries(uni_simd_pfb_channelizer_tests PRIVATE ${UNI_SIMD_INTERNAL_TARGET})
    target_include_directories(uni_simd_pfb_channelizer_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_features(uni_simd_pfb_channelizer_tests PRIVATE cxx_std_23)
    set(UNI_SIMD_TEST_TARGETS uni_simd_tests uni_simd_pfb_channelizer_tests)
    add_executable(uni_simd_ifft_tests src_tests/test_ifft_cf32.cpp)
    target_link_libraries(uni_simd_ifft_tests PRIVATE ${UNI_SIMD_INTERNAL_TARGET})
    target_include_directories(uni_simd_ifft_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_features(uni_simd_ifft_tests PRIVATE cxx_std_23)
    list(APPEND UNI_SIMD_TEST_TARGETS uni_simd_ifft_tests)
    add_test(NAME uni_simd_ifft_tests COMMAND uni_simd_ifft_tests)
    add_executable(uni_simd_c_api_tests src_tests/test_c_api.c)
    target_link_libraries(uni_simd_c_api_tests PRIVATE Uni::SIMD)
    set_target_properties(uni_simd_c_api_tests PROPERTIES LINKER_LANGUAGE CXX)
    list(APPEND UNI_SIMD_TEST_TARGETS uni_simd_c_api_tests)
    if(UNI_SIMD_ENABLE_SANITIZERS)
        foreach(UNI_SIMD_TEST_TARGET IN LISTS UNI_SIMD_TEST_TARGETS)
            target_compile_options(${UNI_SIMD_TEST_TARGET} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${UNI_SIMD_TEST_TARGET} PRIVATE -fsanitize=address,undefined)
        endforeach()
    endif()
    add_test(NAME uni_simd_tests COMMAND uni_simd_tests)
    add_test(NAME uni_simd_pfb_channelizer_tests COMMAND uni_simd_pfb_channelizer_tests)
    add_test(NAME uni_simd_c_api_tests COMMAND uni_simd_c_api_tests)
endif()

if(UNI_SIMD_BUILD_BENCHMARKS)
    add_subdirectory(src_benchmark)
endif()

if(PROJECT_IS_TOP_LEVEL)
    configure_package_config_file(
        cmake/UniSIMDConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/UniSIMDConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/UniSIMD")
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/UniSIMDConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMinorVersion)
    install(TARGETS UniSIMD EXPORT UniSIMDTargets
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
    install(FILES
            include/uni_simd.h
            include/uni_simd_kernels.h
            include/uni_simd_typedefs.h
            "${CMAKE_CURRENT_BINARY_DIR}/generated/uni_simd_export.h"
            "${CMAKE_CURRENT_BINARY_DIR}/generated/uni_simd_version.h"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/uni/simd")
    install(EXPORT UniSIMDTargets FILE UniSIMDTargets.cmake NAMESPACE Uni:: DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/UniSIMD")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/UniSIMDConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/UniSIMDConfigVersion.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/UniSIMD")
endif()
