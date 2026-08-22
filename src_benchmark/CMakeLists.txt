find_package(Threads REQUIRED)
find_package(fmt CONFIG REQUIRED)

add_executable(uni_simd_benchmark benchmark_simd.cpp cpu_topology.cpp)
target_link_libraries(uni_simd_benchmark PRIVATE Uni::SIMD Threads::Threads)

add_executable(uni_simd_benchmark_multi_channel_ddc benchmark_multi_channel_ddc.cpp cpu_topology.cpp)
target_link_libraries(uni_simd_benchmark_multi_channel_ddc PRIVATE Uni::SIMD Threads::Threads)
if(WIN32)
    target_link_libraries(uni_simd_benchmark_multi_channel_ddc PRIVATE PowrProf)
endif()
target_compile_features(uni_simd_benchmark_multi_channel_ddc PRIVATE cxx_std_23)
set_target_properties(uni_simd_benchmark_multi_channel_ddc PROPERTIES CXX_EXTENSIONS OFF)

if(WIN32)
    target_link_libraries(uni_simd_benchmark PRIVATE fmt::fmt-header-only)
else()
    target_link_libraries(uni_simd_benchmark PRIVATE fmt::fmt)
endif()

if(WIN32)
    target_link_libraries(uni_simd_benchmark PRIVATE PowrProf)
endif()
target_compile_features(uni_simd_benchmark PRIVATE cxx_std_23)
set_target_properties(uni_simd_benchmark PROPERTIES CXX_EXTENSIONS OFF)

if(UNI_SIMD_ENABLE_SANITIZERS)
    target_compile_options(uni_simd_benchmark PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(uni_simd_benchmark PRIVATE -fsanitize=address,undefined)
    target_compile_options(uni_simd_benchmark_multi_channel_ddc PRIVATE
                           -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(uni_simd_benchmark_multi_channel_ddc PRIVATE -fsanitize=address,undefined)
endif()
