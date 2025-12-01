cmake_minimum_required(VERSION 3.10)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SENTENCEPIECE QUIET sentencepiece)
if(NOT SENTENCEPIECE_FOUND)
    message(WARNING "SentencePiece not found via pkg-config; continuing without it. If you need tokenizer support, install sentencepiece and re-run CMake.")
    set(SENTENCEPIECE_INCLUDE_DIRS "")
    set(SENTENCEPIECE_LIBRARIES "")
endif()

add_library(t5_common STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/weight_loader.cpp
)

target_include_directories(t5_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${SENTENCEPIECE_INCLUDE_DIRS}
)

# Allow common headers to reference version1 headers (some headers include stack.hpp)
target_include_directories(t5_common PUBLIC
    ${CMAKE_SOURCE_DIR}/version1_plainC/include
)

target_link_libraries(t5_common PUBLIC
    ${SENTENCEPIECE_LIBRARIES}
)


find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(PY_VENV_DIR ${CMAKE_SOURCE_DIR}/.venv)
set(PY_VENV_BIN ${PY_VENV_DIR}/bin)



add_custom_target(venv
    COMMAND ${Python3_EXECUTABLE} -m venv ${PY_VENV_DIR}
    COMMENT "Create Python venv at ${PY_VENV_DIR}"
)

add_custom_target(pip_install
    COMMAND ${PY_VENV_BIN}/pip install -r ${CMAKE_SOURCE_DIR}/common/python/requirements.txt
    DEPENDS venv
    COMMENT "Install Python requirements into ${PY_VENV_DIR}"
)

add_custom_target(generate-test-cases
    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR} ${PY_VENV_BIN}/python ${CMAKE_SOURCE_DIR}/common/python/t5_testkit/main.py
    DEPENDS pip_install
)




