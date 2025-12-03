# Minimal convenience Makefile that wraps CMake targets

.PHONY: help configure build generate generate-fast venv pip_install clean

help:
	@printf "Targets:\n  configure       Configure the build directory\n  build           Build default target\n  generate        Create .venv, install deps and run generator\n  generate-fast   Run generator using system Python (no venv)\n  venv            Create .venv via CMake\n  pip_install     Install python requirements into .venv via CMake\n  clean           Remove build, .venv and generated test_cases\n"

configure:
	@mkdir -p build && cmake -S . -B build

build: configure
	@cmake --build build -- -j$(shell nproc)

generate: configure
	@cmake --build build --target generate-test-cases


venv: configure
	@cmake --build build --target venv

pip_install: configure
	@cmake --build build --target pip_install


clean:
	@rm -rf build .venv test_cases