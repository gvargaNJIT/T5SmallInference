cmake_minimum_required(VERSION 3.10)
project(T5_Inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SentencePiece
find_package(PkgConfig REQUIRED)
pkg_check_modules(SENTENCEPIECE REQUIRED sentencepiece)

# Common library
add_library(t5_common STATIC
    common/src/config.cpp
    common/src/tokenizer.cpp
    common/src/weight_loader.cpp
)

target_include_directories(t5_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include
    ${SENTENCEPIECE_INCLUDE_DIRS}
)

target_link_libraries(t5_common PUBLIC
    ${SENTENCEPIECE_LIBRARIES}
)

# Version 1: Serial CPU implementation
add_library(t5_serial STATIC
    version1_plainC/src/attention.cpp
    version1_plainC/src/block.cpp
    version1_plainC/src/embedding.cpp
    version1_plainC/src/feedforward.cpp
    version1_plainC/src/layer_norm.cpp
    version1_plainC/src/matrix_ops.cpp
    version1_plainC/src/serial_model.cpp
    version1_plainC/src/stack.cpp
)

target_include_directories(t5_serial PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/version1_plainC/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include
)

target_link_libraries(t5_serial PUBLIC
    t5_common
)

# Main executable for Version 1
add_executable(t5_v1_main
    version1_plainC/main.cpp
)

target_link_libraries(t5_v1_main PRIVATE
    t5_serial
    t5_common
)

# Compiler optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(t5_serial PRIVATE -O3 -march=native)
    target_compile_options(t5_v1_main PRIVATE -O3 -march=native)
endif()