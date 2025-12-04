cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-14)

project(t5_cuda LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 50)

file(GLOB_RECURSE CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE CU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

add_executable(t5_cuda
    main.cpp
    ${CPP_SOURCES}
    ${CU_SOURCES}
)

target_include_directories(t5_cuda PRIVATE
    ${MPI_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common/include
)

target_link_libraries(t5_cuda
    CUDA::cudart
    t5_common
)

set_target_properties(t5_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)