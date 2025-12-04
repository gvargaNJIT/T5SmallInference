cmake_minimum_required(VERSION 3.24)

message(STATUS "Configuring version2_mpiOnly (MPI) targets")

project(t5_mpi LANGUAGES CXX)
find_package(MPI REQUIRED)

file(GLOB_RECURSE CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")


add_executable(t5_mpi
    main.cpp
    ${CPP_SOURCES}
)

target_include_directories(t5_mpi PRIVATE
    ${MPI_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common/include
)

target_link_libraries(t5_mpi
    MPI::MPI_CXX
    t5_common
)
