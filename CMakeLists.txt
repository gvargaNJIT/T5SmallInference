cmake_minimum_required(VERSION 3.18)

# Force GCC 14 before defining the project
set(CMAKE_C_COMPILER /usr/bin/gcc-14)
set(CMAKE_CXX_COMPILER /usr/bin/g++-14)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-14)

project(rms_test LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Set your GPU architecture
set(CMAKE_CUDA_ARCHITECTURES 50)

# Compile everything together - no separate compilation
add_executable(test_rms_both
    test_layer_norm_mpi.cpp
    version4_mpi_cuda/src/rms_norm.cu
)

target_include_directories(test_rms_both PRIVATE
    ${MPI_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include
)

target_link_libraries(test_rms_both
    MPI::MPI_CXX
    CUDA::cudart
)

# Important: Turn OFF separate compilation
set_target_properties(test_rms_both PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)