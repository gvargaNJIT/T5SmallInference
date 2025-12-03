cmake_minimum_required(VERSION 3.10)

message(STATUS "Configuring version2_mpiOnly (MPI) targets")

find_package(MPI REQUIRED)

file(GLOB MPI_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB MPI_TENSOR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/*.cpp")
file(GLOB MPI_ATTENTION_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/attention/*.cpp")

set(ALL_MPI_SOURCES ${MPI_SOURCES} ${MPI_TENSOR_SOURCES} ${MPI_ATTENTION_SOURCES})

add_library(t5_mpi_lib STATIC
    ${ALL_MPI_SOURCES}
)

target_include_directories(t5_mpi_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common/include
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(t5_mpi_lib PUBLIC
    t5_common
    ${MPI_CXX_LIBRARIES}
)

add_executable(t5_mpi
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_include_directories(t5_mpi PRIVATE
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(t5_mpi PRIVATE
    t5_common
    t5_mpi_lib
    ${MPI_CXX_LIBRARIES}
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(t5_mpi_lib PRIVATE -O3 -march=native)
    target_compile_options(t5_mpi PRIVATE -O3 -march=native)
endif()