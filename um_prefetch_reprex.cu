/*
 * Copyright (c) 2019, German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the <organization> nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// If defined, use CUDA managed memory, otherwise allocate with malloc
#define USE_MANAGED

// If defined, then set the memAdviseSetAccessedBy flag
// #define ADVISE_ACCESSED_BY

// If defined, then set the memAdviseSetReadMostly flag
// #define ADVISE_READ_MOSTLY

// #define ADVISE_PREFERRED_LOCATION_CPU

// If defined, then touch the data on the host between kernel launches to avoid
// device-side caching
// #define TOUCH_ON_HOST

// If defined, then read data on GPU, else write data on GPU
#define OP_READ

// 32 GiB of data
constexpr unsigned long long SIZE = 32 * 1024 * 1024 * (1024 / sizeof(int));

// Prefetch data in 16 MiB blocks
constexpr unsigned long long PREFETCH_SIZE = 16 * 1024 * 1024 / sizeof(int);

// Number of runs
constexpr unsigned RUNS = 5;

// Device
constexpr int DEVICE_ID = 0;

// NUMA node
constexpr int NUMA_NODE = 0;

#ifndef USE_MANAGED
#include <cstdlib>
#endif

#include <algorithm>
#include <iostream>
#include <utility>
#include <cuda_runtime.h>
#include <cstdint>
#include <numa.h>

#define CHECK_CUDA(ans) check_cuda((ans), __FILE__, __LINE__)
void check_cuda(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
      std::cerr
          << "Exit with code "
          << cudaGetErrorString(code)
          << " (" << code << ") "
          << "in file " << file << ":" << line
          << std::endl;
      std::exit(1);
    }
}

__global__ void read_kernel(int *data, uint64_t len, int *result) {
    const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned gstride = blockDim.x * gridDim.x;

    int counter = 0;
    for (uint64_t i = gid; i < len; i += gstride) {
        counter += data[i];
    }

    atomicAdd(result, counter);
}

__global__ void write_kernel(int *data, uint64_t len) {
    const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned gstride = blockDim.x * gridDim.x;

    for (uint64_t i = gid; i < len; i += gstride) {
        data[i] = i;
    }
}

int main() {
    // Kernel launch parameters
    int sm_count = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, DEVICE_ID));
    int warp_size = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, DEVICE_ID));
    const unsigned GRID_DIM = sm_count * 2;
    const unsigned BLOCK_DIM = warp_size * 4;

    std::cout
        << "Running on device " << DEVICE_ID
        << " with grid dim " << GRID_DIM
        << " and block dim " << BLOCK_DIM
        << " and prefetching " << SIZE / PREFETCH_SIZE << " data blocks"
        << std::endl;

    // Set CUDA device
    CHECK_CUDA(cudaSetDevice(DEVICE_ID));

    // Set NUMA node
    numa_run_on_node(NUMA_NODE);

    // Allocate managed memory
    int *data = nullptr;
#ifdef USE_MANAGED
    CHECK_CUDA(cudaMallocManaged(&data, SIZE * sizeof(int)));
    std::cout << "Managed memory enabled" << std::endl;

#ifdef ADVISE_READ_MOSTLY
    CHECK_CUDA(cudaMemAdvise(data, SIZE * sizeof(int), cudaMemAdviseSetReadMostly, DEVICE_ID));
    std::cout << "cudaMemAdviseSetReadMostly enabled" << std::endl;
#endif

#ifdef ADVISE_ACCESSED_BY
    CHECK_CUDA(cudaMemAdvise(data, SIZE * sizeof(int), cudaMemAdviseSetAccessedBy, DEVICE_ID));
    std::cout << "cudaMemAdviseSetAccessedBy enabled" << std::endl;
#endif

#ifdef ADVISE_PREFERRED_LOCATION_CPU
    CHECK_CUDA(cudaMemAdvise(data, SIZE * sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    std::cout << "cudaMemAdviseSetPreferredLocation CPU enabled" << std::endl;
#endif

#else
    data = (int*) numa_alloc_onnode(SIZE * sizeof(int), NUMA_NODE);
    std::cout << "System memory enabled" << std::endl;
#endif

#ifdef TOUCH_ON_HOST
    std::cout << "Touch on host between runs enabled" << std::endl;
#endif

    // Fill data array
    for (uint64_t i = 0; i < SIZE; ++i) {
        data[i] = i;
    }

    // Allocate result
    int *result = nullptr;
    CHECK_CUDA(cudaMalloc(&result, sizeof(int)));

    // Setup events
    cudaEvent_t start_timer[2], end_timer[2];
    CHECK_CUDA(cudaEventCreate(&start_timer[0]));
    CHECK_CUDA(cudaEventCreate(&start_timer[1]));
    CHECK_CUDA(cudaEventCreate(&end_timer[0]));
    CHECK_CUDA(cudaEventCreate(&end_timer[1]));

    // Setup streams
    cudaStream_t stream[2];
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream[0], cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream[1], cudaStreamNonBlocking));

#ifdef OP_READ
    std::cout << "Running read kernel" << std::endl;
#else
    std::cout << "Running write kernel" << std::endl;
#endif

    for (unsigned run = 0; run < RUNS; ++run) {
        CHECK_CUDA(cudaEventRecord(start_timer[0], stream[0]));
        CHECK_CUDA(cudaEventRecord(start_timer[1], stream[1]));

        for (uint64_t i = 0; i < (SIZE / PREFETCH_SIZE); ++i) {
            // Prefetch some data and launch kernel, and measure time
            CHECK_CUDA(cudaMemPrefetchAsync(&data[i * PREFETCH_SIZE], PREFETCH_SIZE * sizeof(int), DEVICE_ID, stream[0]));
#ifdef OP_READ
            read_kernel<<<GRID_DIM, BLOCK_DIM, 0, stream[0]>>>(&data[i * PREFETCH_SIZE], PREFETCH_SIZE, result);
#else
            write_kernel<<<GRID_DIM, BLOCK_DIM, 0, stream[0]>>>(&data[i * PREFETCH_SIZE], PREFETCH_SIZE);
#endif

            // Swap stream and keep track of which events track which stream
            std::swap(stream[0], stream[1]);
            std::swap(start_timer[0], start_timer[1]);
            std::swap(end_timer[0], end_timer[1]);
        }

        CHECK_CUDA(cudaEventRecord(end_timer[0], stream[0]));
        CHECK_CUDA(cudaEventRecord(end_timer[1], stream[1]));

        // Wait for kernel completion
        CHECK_CUDA(cudaDeviceSynchronize());

        // Calculate the time taken as maximum of all timer combinations
        float stream_time_ms[4] = {0.0};
        CHECK_CUDA(cudaEventElapsedTime(&stream_time_ms[0], start_timer[0], end_timer[0]));
        CHECK_CUDA(cudaEventElapsedTime(&stream_time_ms[1], start_timer[0], end_timer[1]));
        CHECK_CUDA(cudaEventElapsedTime(&stream_time_ms[2], start_timer[1], end_timer[0]));
        CHECK_CUDA(cudaEventElapsedTime(&stream_time_ms[3], start_timer[1], end_timer[1]));
        float time_ms = std::max(stream_time_ms[0], stream_time_ms[1]);
        time_ms = std::max(time_ms, stream_time_ms[2]);
        time_ms = std::max(time_ms, stream_time_ms[3]);

        // Compute and print throughput in GiB/s
        uint64_t size_GiB = (SIZE * sizeof(int)) / 1024 / 1024 / 1024;
        double tput = ((double)size_GiB) / ((double)time_ms) * 1000.0;
        std::cout << "Throughput: " << tput << " GiB/s" << std::endl;

#ifdef TOUCH_ON_HOST
        for (uint64_t i = 0; i < SIZE; ++i) {
            data[i] = run + i;
        }
#endif
    }

    // Cleanup
    CHECK_CUDA(cudaStreamDestroy(stream[0]));
    CHECK_CUDA(cudaStreamDestroy(stream[1]));
    CHECK_CUDA(cudaEventDestroy(start_timer[0]));
    CHECK_CUDA(cudaEventDestroy(start_timer[1]));
    CHECK_CUDA(cudaEventDestroy(end_timer[0]));
    CHECK_CUDA(cudaEventDestroy(end_timer[1]));
#ifdef USE_MANAGED
    CHECK_CUDA(cudaFree(data));
#else
    numa_free(data, SIZE * sizeof(int));
#endif
    CHECK_CUDA(cudaFree(result));
}
