cmake_minimum_required(VERSION 3.10)

message(STATUS "Configuring version1_plainC (serial) targets")

file(GLOB SERIAL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(t5_serial_lib STATIC
    ${SERIAL_SOURCES}
)

target_include_directories(t5_serial_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common/include
)

target_link_libraries(t5_serial_lib PUBLIC
    t5_common
)

add_executable(t5_serial
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_link_libraries(t5_serial PRIVATE
    t5_serial_lib
    t5_common
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(t5_serial_lib PRIVATE -O3 -march=native)
    target_compile_options(t5_serial PRIVATE -O3 -march=native)
endif()
