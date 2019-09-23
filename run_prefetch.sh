#!/usr/bin/env bash

set -e

nvcc -std=c++11 -O3 -lnuma -gencode arch=compute_70,code=sm_70 -o um_prefetch_reprex um_prefetch_reprex.cu
./um_prefetch_reprex
