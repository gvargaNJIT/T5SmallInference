cmake_minimum_required(VERSION 3.10)

message(STATUS "Configuring tests targets")

# ----------------------------------------------------
# Find MPI
# ----------------------------------------------------
find_package(MPI REQUIRED)

# ----------------------------------------------------
# Build a minimal tensor library for tests
# ----------------------------------------------------
add_library(t5_tensor STATIC
    ${CMAKE_SOURCE_DIR}/version1_plainC/src/tensor.cpp
)

target_include_directories(t5_tensor PUBLIC
    ${CMAKE_SOURCE_DIR}/common/include
    ${CMAKE_SOURCE_DIR}/version1_plainC/include
)

# ----------------------------------------------------
# Build the MPI backend library
# (adjust the filenames if your MPI backend has more files)
# ----------------------------------------------------
add_library(version2_mpi STATIC
    ${CMAKE_SOURCE_DIR}/version2_mpiOnly/src/tensor.cpp
)

target_include_directories(version2_mpi PUBLIC
    ${CMAKE_SOURCE_DIR}/common/include
    ${CMAKE_SOURCE_DIR}/version2_mpiOnly/include
)

target_link_libraries(version2_mpi  PUBLIC MPI::MPI_CXX)

# ----------------------------------------------------
# Build the test executable
# ----------------------------------------------------
add_executable(test_tensor
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tensor.cpp
)

target_include_directories(test_tensor PRIVATE
    ${CMAKE_SOURCE_DIR}/common/include
    ${CMAKE_SOURCE_DIR}/version1_plainC/include
    ${CMAKE_SOURCE_DIR}/version2_mpi/include
)

target_link_libraries(test_tensor PRIVATE
    t5_tensor
    version2_mpi 
    MPI::MPI_CXX
)

# ----------------------------------------------------
# Optional optimizations
# ----------------------------------------------------
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(test_tensor PRIVATE -O3 -march=native)
endif()
