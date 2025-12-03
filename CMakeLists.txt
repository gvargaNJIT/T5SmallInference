cmake_minimum_required(VERSION 3.10)
project(T5_Inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(common)
add_subdirectory(version1_plainC)
add_subdirectory(version2_mpiOnly)
add_subdirectory(version3_cudaOnly)
add_subdirectory(version4_mpi_cuda)
add_subdirectory(tests)
add_subdirectory(testing)


