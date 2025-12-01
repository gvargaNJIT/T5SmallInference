cmake_minimum_required(VERSION 3.18)

# Force GCC 14 before defining the project
set(CMAKE_C_COMPILER /usr/bin/gcc-14)
set(CMAKE_CXX_COMPILER /usr/bin/g++-14)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-14)

project(embedding_test LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Set your GPU architecture
set(CMAKE_CUDA_ARCHITECTURES 50)

# Common include directory
set(COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/common/include)

# ============================================================================
# EMBEDDING TESTS
# ============================================================================

# Serial CPU Version
add_executable(test_embedding_serial
    test_embedding.cpp
    version1_plainC/src/embedding.cpp
)

target_include_directories(test_embedding_serial PRIVATE
    ${COMMON_INCLUDE}
)

# CUDA Version (no MPI)
add_executable(test_embedding_cuda
    test_embedding.cpp
    version3_cudaOnly/src/embedding.cu
)

target_include_directories(test_embedding_cuda PRIVATE
    ${COMMON_INCLUDE}
)

target_link_libraries(test_embedding_cuda
    CUDA::cudart
)

set_target_properties(test_embedding_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# MPI Version (CPU only)
add_executable(test_embedding_mpi
    test_embedding_mpi.cpp
    version2_mpiOnly/src/embedding.cpp
)

target_include_directories(test_embedding_mpi PRIVATE
    ${MPI_INCLUDE_PATH}
    ${COMMON_INCLUDE}
)

target_link_libraries(test_embedding_mpi
    MPI::MPI_CXX
)

# MPI + CUDA Version
add_executable(test_embedding_mpi_cuda
    test_embedding_mpi.cpp
    version4_mpi_cuda/src/embedding.cu
)

target_include_directories(test_embedding_mpi_cuda PRIVATE
    ${MPI_INCLUDE_PATH}
    ${COMMON_INCLUDE}
)

target_link_libraries(test_embedding_mpi_cuda
    MPI::MPI_CXX
    CUDA::cudart
)

set_target_properties(test_embedding_mpi_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ============================================================================
# RMS NORM TESTS
# ============================================================================

# Serial CPU Version
add_executable(test_rmsnorm_serial
    test_layer_norm.cpp
    version1_plainC/src/rms_norm.cpp
)

target_include_directories(test_rmsnorm_serial PRIVATE
    ${COMMON_INCLUDE}
)

# CUDA Version (no MPI)
add_executable(test_rmsnorm_cuda
    test_layer_norm.cpp
    version3_cudaOnly/src/rms_norm.cu
)

target_include_directories(test_rmsnorm_cuda PRIVATE
    ${COMMON_INCLUDE}
)

target_link_libraries(test_rmsnorm_cuda
    CUDA::cudart
)

set_target_properties(test_rmsnorm_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# MPI Version (CPU only)
add_executable(test_rmsnorm_mpi
    test_layer_norm_mpi.cpp
    version2_mpiOnly/src/rms_norm.cpp
)

target_include_directories(test_rmsnorm_mpi PRIVATE
    ${MPI_INCLUDE_PATH}
    ${COMMON_INCLUDE}
)

target_link_libraries(test_rmsnorm_mpi
    MPI::MPI_CXX
)

# MPI + CUDA Version
add_executable(test_rmsnorm_mpi_cuda
    test_layer_norm_cudampi.cpp
    version4_mpi_cuda/src/rms_norm.cu
)

target_include_directories(test_rmsnorm_mpi_cuda PRIVATE
    ${MPI_INCLUDE_PATH}
    ${COMMON_INCLUDE}
)

target_link_libraries(test_rmsnorm_mpi_cuda
    MPI::MPI_CXX
    CUDA::cudart
)

set_target_properties(test_rmsnorm_mpi_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)