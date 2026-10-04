cmake_minimum_required(VERSION 3.16)
project(inferno LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(INFERNO_CUDA "Compile CUDA kernels and CUDA graph capture" OFF)

find_package(Threads REQUIRED)
find_package(OpenMP)

add_library(inferno
    src/cpu_kernels.cpp
    src/paged_cache.cpp
    src/model.cpp
    src/scheduler.cpp
    src/engine.cpp
    src/speculative.cpp
    src/server.cpp
    src/bench.cpp
)
target_include_directories(inferno PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(inferno PUBLIC Threads::Threads)
if(OpenMP_CXX_FOUND)
    target_link_libraries(inferno PUBLIC OpenMP::OpenMP_CXX)
endif()

if(INFERNO_CUDA)
    enable_language(CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    target_sources(inferno PRIVATE kernels/cuda_kernels.cu)
    target_compile_definitions(inferno PUBLIC INFERNO_USE_CUDA)
    set_property(TARGET inferno PROPERTY CUDA_STANDARD 17)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_link_libraries(inferno PUBLIC CUDA::cublas CUDA::cudart)
    else()
        target_link_libraries(inferno PUBLIC cublas cudart)
    endif()
else()
    target_sources(inferno PRIVATE src/cuda_stub.cpp)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(inferno PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>)
endif()

add_executable(inferno_cli src/main.cpp)
target_link_libraries(inferno_cli PRIVATE inferno)
set_target_properties(inferno_cli PROPERTIES OUTPUT_NAME inferno)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(inferno_cli PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

add_executable(inferno_tests tests/test_inferno.cpp)
target_link_libraries(inferno_tests PRIVATE inferno)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(inferno_tests PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

enable_testing()
add_test(NAME inferno_tests COMMAND inferno_tests)
