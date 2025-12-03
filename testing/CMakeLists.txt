cmake_minimum_required(VERSION 3.10)

# Common test runner library
add_library(test_common STATIC
    common/test_runner.cpp
)

target_include_directories(test_common PUBLIC
    ${CMAKE_SOURCE_DIR}/common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

# Serial test executable
add_executable(test_serial
    serial/test_serial.cpp
    serial/serial_test_runner.cpp
)

target_include_directories(test_serial PRIVATE
    ${CMAKE_SOURCE_DIR}/common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/serial
)

target_link_libraries(test_serial
    test_common
    t5_serial_lib
)

# MPI test executable
find_package(MPI REQUIRED)

add_executable(test_mpi
    mpi/test_mpi.cpp
    mpi/mpi_test_runner.cpp
)

target_include_directories(test_mpi PRIVATE
    ${CMAKE_SOURCE_DIR}/common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/mpi
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(test_mpi
    test_common
    t5_mpi_lib
    ${MPI_CXX_LIBRARIES}
)

if(MPI_CXX_COMPILE_FLAGS)
    set_target_properties(test_mpi PROPERTIES
        COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
endif()

if(MPI_CXX_LINK_FLAGS)
    set_target_properties(test_mpi PROPERTIES
        LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
endif()

message(STATUS "Testing framework configured:")
message(STATUS "  - test_serial: Serial tensor tests")
message(STATUS "  - test_mpi: MPI parallel tensor tests")
