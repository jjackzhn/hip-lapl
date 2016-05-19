cuda-lapl
==========

=== Forked to use with HIP tools ===

This is a benchmark that performs a discreet 2D laplacian operation a number of times on a given vector. The benchmark therefore evaluates the performance of nearest neighboring single hop stencil operations.

The benchmark has been optimised for NVIDIA GPUs using CUDA. The CUDA version is compared to a version using SSE intrinsics which runs on the CPU.

Python files are included to create an initial vector and to visualize the resulting vector. The application can be thought of as evolving the 2D heat equation on a given initial heat distribution.

The benchmark code was developed under the Cyprus Research Promotion Foundation (RPF) project "GPU Clusterware", Grant Number: ΤΠΕ/ΠΛΗΡΟ/0311(ΒΙΕ)/09
