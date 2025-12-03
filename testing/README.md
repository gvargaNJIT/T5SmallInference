# Testing Framework

This directory contains a separate, scalable testing framework for both serial (version1_plainC) and MPI (version2_mpi) tensor implementations.

## Structure

```
testing/
├── common/              # Shared test infrastructure
│   ├── test_runner.hpp  # Abstract base class
│   └── test_runner.cpp  # Test loading and comparison logic
├── serial/              # Serial tests
│   ├── serial_test_runner.hpp
│   ├── serial_test_runner.cpp
│   └── test_serial.cpp  # Main executable
├── mpi/                 # MPI tests
│   ├── mpi_test_runner.hpp
│   ├── mpi_test_runner.cpp
│   └── test_mpi.cpp     # Main executable
└── CMakeLists.txt       # Build configuration
```

## Building

From the project root:

```bash
mkdir -p build && cd build
cmake ..
make test_serial    # Build serial tests
make test_mpi       # Build MPI tests
```

## Running Tests

### Serial Tests

```bash
# Run from build directory with default test location
./testing/test_serial

# Or specify custom test directory
./testing/test_serial /path/to/test_cases/tensor
```

### MPI Tests

```bash
# Run with single process
mpirun -n 1 ./testing/test_mpi

# Run with multiple processes (4 in this example)
mpirun -n 4 ./testing/test_mpi

# With custom test directory
mpirun -n 4 ./testing/test_mpi /path/to/test_cases/tensor
```

## Test Case Format

Test cases are binary files located in `test_cases/tensor/` with naming convention:
- `operation_testnum.bin` (e.g., `matmul_1.bin`, `softmax_0.bin`)

Each file contains:
- For binary ops (matmul, add): 2 input tensors + 1 expected output tensor
- For unary ops (softmax, permute): 1 input tensor + 1 expected output tensor

Binary format per tensor:
1. Name length (int32)
2. Name string
3. Number of dimensions (int32)
4. Shape array (int32 × ndim)
5. Flat data length (int32)
6. Data array (float32 × flat_len)

## Supported Operations

- `matmul`: Matrix multiplication
- `add`: Element-wise addition
- `softmax`: Softmax activation
- `permute`: Tensor permutation/transpose

## Exit Codes

- `0`: All tests passed
- `1`: One or more tests failed

## Implementation Details

### Common Base Class

`TestRunner` provides:
- Binary test file loading
- Test execution framework
- Tensor comparison with tolerance (default 1e-5)
- Pass/fail tracking and reporting

### Serial Test Runner

- Directly calls `Tensor` methods from `version1_plainC`
- Single-threaded execution
- No MPI dependencies

### MPI Test Runner

- Uses MPI-parallel tensor operations from `version2_mpi`
- All ranks participate in operations
- Only rank 0 prints results
- Test data loaded on rank 0 only

## Adding New Operations

1. Add test case generation in Python test kit
2. Update `execute_operation()` in both serial and MPI test runners
3. Implement the operation in respective Tensor classes

## Troubleshooting

**No tests found:**
- Check that test case directory exists and contains `.bin` files
- Verify path is correct (relative to build directory)

**Shape mismatch errors:**
- Verify test case generation matches tensor implementation
- Check that dimensions are consistent

**MPI hangs:**
- Ensure all ranks participate in collective operations
- Check that input tensors are properly distributed
