cmake_minimum_required(VERSION 3.18)

if(EXISTS /usr/bin/gcc-14)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-14)
endif()

project(t5_mpicuda LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "50;75;80;86")
endif()



file(GLOB_RECURSE CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE CU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

add_executable(t5_mpicuda
    main.cpp
    ${CPP_SOURCES}
    ${CU_SOURCES}
)

target_include_directories(t5_mpicuda PRIVATE
    ${MPI_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common/include
)

target_link_libraries(t5_mpicuda
    MPI::MPI_CXX
    CUDA::cudart
    t5_common
)


set_target_properties(t5_mpicuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)