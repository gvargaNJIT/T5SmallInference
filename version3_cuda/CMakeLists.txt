cmake_minimum_required(VERSION 3.18)

message(STATUS "Configuring version3_cuda (CUDA) targets")

enable_language(CUDA)

file(GLOB CUDA_CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB CUDA_CPP_SOURCES_ATTN "${CMAKE_CURRENT_SOURCE_DIR}/src/attention/*.cpp")
file(GLOB CUDA_TENSOR_CPP "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/*.cpp")
file(GLOB CUDA_KERNEL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/*.cu")
file(GLOB CUDA_ATTN_KERNEL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/attention/*.cu")

set(ALL_CUDA_SOURCES ${CUDA_CPP_SOURCES} ${CUDA_CPP_SOURCES_ATTN} ${CUDA_TENSOR_CPP} ${CUDA_KERNEL_SOURCES} ${CUDA_ATTN_KERNEL_SOURCES})

add_library(t5_cuda_lib STATIC
    ${ALL_CUDA_SOURCES}
)

target_include_directories(t5_cuda_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/version1_plainC/include
    ${CMAKE_SOURCE_DIR}/common/include
)

target_link_libraries(t5_cuda_lib PUBLIC
    t5_common
)

set_target_properties(t5_cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86"
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
    add_executable(t5_cuda
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    )

    target_link_libraries(t5_cuda PRIVATE
        t5_cuda_lib
        t5_common
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(t5_cuda_lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>)
        target_compile_options(t5_cuda PRIVATE -O3 -march=native)
    endif()
endif()
