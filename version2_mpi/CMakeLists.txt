cmake_minimum_required(VERSION 3.10)

message(STATUS "Configuring version2_mpi (MPI parallel) targets")

find_package(MPI REQUIRED)

file(GLOB MPI_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

file(GLOB MPI_TENSOR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/*.cpp")

set(ALL_MPI_SOURCES ${MPI_SOURCES} ${MPI_TENSOR_SOURCES})

add_library(t5_mpi_lib STATIC
    ${ALL_MPI_SOURCES}
)

target_include_directories(t5_mpi_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/version1_plainC/include
    ${CMAKE_SOURCE_DIR}/common/include
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(t5_mpi_lib PUBLIC
    t5_common
    MPI::MPI_CXX
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
    add_executable(t5_mpi
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    )

    target_link_libraries(t5_mpi PRIVATE
        t5_mpi_lib
        t5_common
        MPI::MPI_CXX
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(t5_mpi_lib PRIVATE -O3 -march=native)
        target_compile_options(t5_mpi PRIVATE -O3 -march=native)
    endif()
endif()
