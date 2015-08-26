# MPI Parallel

This branch provides data parallelization for Caffe based on MPI.

## Installation

- Install `openmpi` with `apt-get`, or `pacman`, or `yum`, etc.
- Uncomment the MPI parallel block in the Makefile.config and set the `MPI_INCLUDE` and `MPI_LIB` correspondingly.
- `make clean && make -j`

## Usage

You don't need to change your prototxt. Just provide the GPU ids in the `-gpu` option (separated by commas). For example:

    mpirun -n 2 build/tools/caffe train \
      -solver examples/mnist/lenet_solver.prototxt \
      -gpu 0,1
