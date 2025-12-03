cmake_minimum_required(VERSION 3.18)

# set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-14)

project(t5_cuda LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 50)

file(GLOB MPICUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB MPICUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

add_library(t5_mpicuda_lib STATIC
    ${MPICUDA_SOURCES}
)

target_include_directories(t5_mpicuda_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common/include
)

target_link_libraries(t5_mpicuda_lib PUBLIC
    t5_common
)

add_executable(t5_mpicuda
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_link_libraries(t5_mpicuda PRIVATE
    t5_mpicuda_lib
    t5_common
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(t5_mpicuda_lib PRIVATE -O3 -march=native)
    target_compile_options(t5_mpicuda PRIVATE -O3 -march=native)
endif()
