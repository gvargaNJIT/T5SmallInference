# T5 - Small Inference Model
### How to compile the code

We used cmake files instead of make files to better adapt to the environment of the older Maxwell Nvidia GPU model in our computers.

This model can only summarize text, translate from English to German, translate from English to French, and translate from English to Romanian.

To compile:

```sh
mkdir build
cmake -S . -B build
```

For all the code:
```sh
cmake --build build
```

For a specific target:
```sh
cmake --build build --target t5_serial
cmake --build build --target t5_mpi
cmake --build build --target t5_cuda
cmake --build build --target t5_mpicuda
```

To run a specific configuration:

Serial:
```sh
./build/version1_plainC/t5_serial
```

MPI:
```sh
mpirun -np 2 --hostfile hostfile.txt ./build/version2_mpiOnly/t5_mpi
mpirun -np 4 --hostfile hostfile.txt ./build/version2_mpiOnly/t5_mpi 
mpirun -np 8 --hostfile hostfile.txt ./build/version2_mpiOnly/t5_mpi
```

CUDA:
```sh
./build/version3_cudaOnly/t5_cuda
```

MPI+CUDA:
```sh
mpirun -np 2 --hostfile hostfile.txt ./build/version4_mpi_cuda/t5_mpicuda
mpirun -np 4 --hostfile hostfile.txt ./build/version4_mpi_cuda/t5_mpicuda
mpirun -np 8 --hostfile hostfile.txt ./build/version4_mpi_cuda/t5_mpicuda
```

To have a custom input:
```sh
./build/<version>/<executable> "custom input"
```
Inputs can only start with:
- "summarize:"
- "translate English to German:"
- "translate English to Romanian:"
- "translate English to French:"

Example inputs:
```sh
./build/version1_plainC/t5_serial  "translate from English to French: The quick brown fox jumps over the lazy dog."

mpirun -np 4 --hostfile hostfile.txt ./build/version2_mpiOnly/t5_mpi "summarize: Einstein captured the world’s imagination with his blend of brilliant scientific theories and humanitarian concern."
```


To remove build folder contents:
```sh
rm -rf build/*
```