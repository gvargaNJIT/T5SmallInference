# Makefile for T5 Inference Project

CXX = g++
NVCC = nvcc
MPICC = mpic++
CXXFLAGS = -std=c++17 -O3 -Wall
NVCCFLAGS = -std=c++17 -O3
MPIFLAGS = -std=c++17 -O3

# Directories
BUILD_DIR = build
COMMON_DIR = common
V1_DIR = version1_plainC
V2_DIR = version2_mpiOnly
V3_DIR = version3_cudaOnly
V4_DIR = version4_mpi_cuda

# Default target
.PHONY: all
all: common version1 version2 version3 version4

# Build common library (if exists)
.PHONY: common
common:
	@if [ -d $(COMMON_DIR) ]; then \
		echo "Building common library..."; \
		$(MAKE) -C $(COMMON_DIR); \
	fi

# Build all versions
.PHONY: version1
version1: common
	@echo "Building Version 1 (Plain C++)..."
	@$(MAKE) -C $(V1_DIR)

.PHONY: version2
version2: common
	@echo "Building Version 2 (MPI Only)..."
	@$(MAKE) -C $(V2_DIR)

.PHONY: version3
version3: common
	@echo "Building Version 3 (CUDA Only)..."
	@$(MAKE) -C $(V3_DIR)

.PHONY: version4
version4: common
	@echo "Building Version 4 (MPI + CUDA)..."
	@$(MAKE) -C $(V4_DIR)

# Clean all builds
.PHONY: clean
clean:
	@echo "Cleaning all builds..."
	@if [ -d $(COMMON_DIR) ]; then $(MAKE) -C $(COMMON_DIR) clean; fi
	@if [ -d $(V1_DIR) ]; then $(MAKE) -C $(V1_DIR) clean; fi
	@if [ -d $(V2_DIR) ]; then $(MAKE) -C $(V2_DIR) clean; fi
	@if [ -d $(V3_DIR) ]; then $(MAKE) -C $(V3_DIR) clean; fi
	@if [ -d $(V4_DIR) ]; then $(MAKE) -C $(V4_DIR) clean; fi
	@rm -rf $(BUILD_DIR)

# Run targets (examples)
.PHONY: run-v1
run-v1: version1
	./$(BUILD_DIR)/$(V1_DIR)/t5_plain

.PHONY: run-v2
run-v2: version2
	mpirun -np 4 ./$(BUILD_DIR)/$(V2_DIR)/t5_mpi

.PHONY: run-v3
run-v3: version3
	./$(BUILD_DIR)/$(V3_DIR)/t5_cuda

.PHONY: run-v4
run-v4: version4
	mpirun -np 4 ./$(BUILD_DIR)/$(V4_DIR)/t5_mpicuda

.PHONY: help
help:
	@echo "T5 Inference Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build all versions (including common library)"
	@echo "  common     - Build common library only"
	@echo "  version1   - Build plain C++ version"
	@echo "  version2   - Build MPI-only version"
	@echo "  version3   - Build CUDA-only version"
	@echo "  version4   - Build MPI+CUDA version"
	@echo "  clean      - Clean all builds"
	@echo "  run-v1     - Run plain C++ version"
	@echo "  run-v2     - Run MPI version (4 processes)"
	@echo "  run-v3     - Run CUDA version"
	@echo "  run-v4     - Run MPI+CUDA version (4 processes)"
	@echo "  help       - Show this help message"