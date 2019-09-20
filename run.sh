#!/usr/bin/env bash

set -e

nvcc -O3 -lnuma -gencode arch=compute_70,code=sm_70 -o um_reprex um_reprex.cu
./um_reprex
