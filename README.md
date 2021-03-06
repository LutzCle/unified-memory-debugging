CUDA Unified Memory Bandwidth Debugging
======================================

This repository contains two minimal, reproducible examples (*reprex*) to debug a performance issue on an IBM AC922 system. One example uses CUDA Unified Memory, and the other uses Unified Memory with asynchronous prefetching.

**Problem description**: Our baseline using NUMA-local host memory performs at 63 GiB/s. Using demand or prefetch paging, we measure only between 1-2 GiB/s on NVLink 2.0. This seems too slow. Measurements on x86-64 show that more bandwidth is possible, even when using the slower PCI-e 3.0 interconnect.

[Link to Nvidia DevTalk post.](https://devtalk.nvidia.com/default/topic/1063552/cuda-programming-and-performance/unified-memory-has-slow-bandwidth-over-nvlink-2-0-for-large-data-sizes/post/5385699/#5385699)

## Insights

So far, we have gained the following insights:

 - Setting `ADVISE_PREFERRED_LOCATION_CPU` circumvents the bandwidth issue by not migrating the pages. Instead, the GPU accesses pages directly in host memory. The first run is slow, because the first run establishes page mappings.
 - Pages can be mapped before the first run by setting `ADVISE_PREFERRED_LOCATION_CPU` and `ADVISE_ACCESSED_BY`. With these settings, we achieve the peak bandwidth.
 - On x86-64, we also achieve peak bandwidth by setting `ADVISE_ACCESSED_BY`.

## Reproducing the problem
After downloading this repository, compile and run the example with:
```sh
./run.sh
```

## Configuration parameters
In the files `um_reprex.cu` and `um_prefetch_reprex.cu`, there are several configuration parameters:
 - USE_MANAGED: If defined, the reprex allocates Unified Memory. Otherwise, NUMA-local system memory is allocated with `numa_alloc_onnode`.
 - ADVISE_ACCESSED_BY: If defined, then set CUDA's memAdviseSetAccessedBy flag on the Unified Memory array.
 - ADVISE_READ_MOSTLY: If defined, then set the memAdviseSetReadMostly flag on the Unified Memory array.
 - ADVISE_PREFERRED_LOCATION_CPU: If defined, then set the cudaMemAdviseSetPreferredLocation flag on the Unified Memory array to resist moving away from the CPU.
 - TOUCH_ON_HOST: If defined, touch the data on the host between kernel runs. This is intended to force cache eviction, because Unified Memory caches data in device memory. Mostly useful for data sizes < 16 GiB.
 - OP_READ: If defined, then read data on GPU (host-to-device), else write data on GPU (device-to-host).
 - SIZE: Data size. Default is 32 GiB.
 - PREFETCH_SIZE: The block size to prefetch at-a-time. Default is 16 MiB.
 - RUNS: Number of kernel runs to measure. Default is 5 runs.
 - DEVICE_ID: CUDA device to run kernels on. Default is device 0.
 - NUMA_NODE: NUMA node to run program and allocate NUMA-local memory on. Default is node 0.

## Measurements
In the following, we demonstrate our performance issue. Unless noted otherwise, we run the example `um_reprex.cu` and conduct our measurments on this system:
 - IBM AC922 with POWER9 and Tesla V100
 - Ubuntu 18.04 LTS ppc64le
 - Kernel 5.0.0-25-generic
 - Nvidia driver 418.67
 - CUDA 10.1.168

Note: We have previously also tried older versions of the Nvidia driver, CUDA, and Linux kernel. So far, we have not found a combination with notably better performance.

### Baseline: NUMA-local memory
**Parameters**: 32 GiB data, `!USE_MANAGED` (i.e., is undefined), `OP_READ`
```sh
Running on device 0 with grid dim 160 and block dim 128
System memory enabled
Running read kernel
Throughput: 63.4921 GiB/s
Throughput: 63.5247 GiB/s
Throughput: 63.5517 GiB/s
Throughput: 63.5563 GiB/s
Throughput: 63.5397 GiB/s
```
**Observations:** 65.65 GiB/s is the theoretical maximum performance of the interconnect. We're measuring the maximum bandwidth we can expect. Best of all, the measurements are repeatable for basically any data size (we tested 1 GiB, 16 GiB, 32 GiB, and 120 GiB).

### Small data: Slow, but expected
**Parameters**: 1 GiB data, `USE_MANAGED`, `OP_READ`
```sh
Running on device 0 with grid dim 160 and block dim 128
Managed memory enabled
Running read kernel
Throughput: 5.23589 GiB/s
Throughput: 207.725 GiB/s
Throughput: 207.317 GiB/s
Throughput: 207.346 GiB/s
Throughput: 207.523 GiB/s
```
**Observations:** First run transfers data, next runs access data that is cached in device memory. Transfer bandwidth is slower than expected, but we expect the caching to increase bandwidth.

### Small data with tuning: Fast, and slightly strange
**Parameters**: 1 GiB data, `USE_MANAGED`, `OP_READ`, `ADVISE_ACCESSED_BY`
```sh
Running on device 0 with grid dim 160 and block dim 128
Managed memory enabled
cudaMemAdviseSetAccessedBy enabled
Running read kernel
Throughput: 53.2082 GiB/s
Throughput: 13.9365 GiB/s
Throughput: 197.721 GiB/s
Throughput: 198.054 GiB/s
Throughput: 198.611 GiB/s
```
**Observations:** Same as before, but setting `ADVISE_ACCESSED_BY` increases transfer bandwidth by 10x on the first run. Strangely, the bandwidth drops on the second run. This is reproducible over multiple executions. From the third run onwards, we again access cached data.

### Large data: Slow as might be expected from before
**Parameters**: 32 GiB data, `USE_MANAGED`, `OP_READ`
```sh
Running on device 0 with grid dim 160 and block dim 128
Managed memory enabled
Running read kernel
Throughput: 1.87802 GiB/s
Throughput: 2.15174 GiB/s
Throughput: 1.49559 GiB/s
Throughput: 1.51264 GiB/s
Throughput: 0.99124 GiB/s
```
**Observations:** With 32 GiB data (i.e., larger than device memory), data cannot be cached anymore. The drop in bandwidth is therefore expected, although the drop is lower than the first run in our first measurement (~2 GiB/s vs. ~5 GiB/s).

### Large data set with tuning: DANGER ZONE
**Parameters**: 32 GiB data, `USE_MANAGED`, `OP_READ`, `ADVISE_ACCESSED_BY`
```sh
Running on device 0 with grid dim 160 and block dim 128
Managed memory enabled
cudaMemAdviseSetAccessedBy enabled
Running read kernel
Throughput: 28.4424 GiB/s
Throughput: 1.8726 GiB/s
Throughput: 1.06627 GiB/s
Throughput: 1.49521 GiB/s
Throughput: 0.94333 GiB/s
```
**Observations:** This is where our problem sets in. The first run transfers data fast-ish. Subsequent runs are much slower than we would like. We expect all runs to have the same bandwidth as the first run. Ideally, bandwidth would equal our baseline at 63 GiB/s.

### Large data with migration resistance: Fast after warm-up
**Parameters**: 32 GiB data, `USE_MANAGED`, `OP_READ`, `ADVISE_PREFERRED_LOCATION_CPU`
```sh
Running on device 0 with grid dim 160 and block dim 128
Managed memory enabled
cudaMemAdviseSetPreferredLocation CPU enabled
Running read kernel
Throughput: 1.11503 GiB/s
Throughput: 63.4069 GiB/s
Throughput: 63.4517 GiB/s
Throughput: 63.4277 GiB/s
Throughput: 63.4083 GiB/s
```
**Observations:** After a slow first run, we see bandwidth that is equal to our baseline.

Thank you to [Robert Crovella on Nvidia DevTalk](https://devtalk.nvidia.com/default/topic/1063552/cuda-programming-and-performance/unified-memory-has-slow-bandwidth-over-nvlink-2-0-for-large-data-sizes/post/5385717/#5385717) for suggesting this option!

### Large data with tuning and migration resistance: Consistently fast
**Parameters**: 32 GiB data, `USE_MANAGED`, `OP_READ`, `ADVISE_ACCESSED_BY`, `ADVISE_PREFERRED_LOCATION_CPU`
```sh
Running on device 0 with grid dim 160 and block dim 128
Managed memory enabled
cudaMemAdviseSetAccessedBy enabled
cudaMemAdviseSetPreferredLocation CPU enabled
Running read kernel
Throughput: 63.4137 GiB/s
Throughput: 63.408 GiB/s
Throughput: 63.4255 GiB/s
Throughput: 63.417 GiB/s
Throughput: 63.3982 GiB/s
```
**Observations:** By setting up the page mappings ahead of the kernel launch in addition to resisting page migration, even the first run is equal to our baseline!

### Large data with prefetching: Slow as without any tuning
**Example**: `um_prefetch_reprex.cu`

**Parameters**: 32 GiB data, `USE_MANAGED`, `OP_READ`
```sh
Running on device 0 with grid dim 160 and block dim 128
Prefetching 16 MiB data blocks
Managed memory enabled
Running read kernel
Throughput: 3.06278 GiB/s
Throughput: 1.69091 GiB/s
Throughput: 1.69134 GiB/s
Throughput: 1.69128 GiB/s
Throughput: 1.69109 GiB/s
```
**Observations:** Prefetching blocks ahead of kernel launch does not improve bandwidth over vanilla Unified Memory.

### Control group: x86-64 with PCI-e
**Parameters**: 32 GiB data, `USE_MANAGED`, `OP_READ`
```sh
Running on device 0 with grid dim 160 and block dim 128
Managed memory enabled
Running read kernel
Throughput: 2.16685 GiB/s
Throughput: 2.17654 GiB/s
Throughput: 2.16583 GiB/s
Throughput: 3.09534 GiB/s
Throughput: 2.15913 GiB/s
```
** Observations**: Similar to the POWER9, on x86 we measure less bandwidth than the maximum of ~11 GiB/s would suggest. What's interesting is that bandwidth tends to be higher than the same benchmark on POWER9 above.

### Control group: x86-64 with tuning
**Parameters**: 32 GiB data, `USE_MANAGED`, `OP_READ`, `ADVISE_ACCESSED_BY`
```sh
Running on device 0 with grid dim 160 and block dim 128
Managed memory enabled
cudaMemAdviseSetAccessedBy enabled
Running read kernel
Throughput: 11.2909 GiB/s
Throughput: 10.8055 GiB/s
Throughput: 10.7137 GiB/s
Throughput: 11.4643 GiB/s
Throughput: 11.0659 GiB/s
```
**Observations**: Unlike on POWER9, setting `ADVISE_ACCESSED_BY` consistently increases bandwidth to essentially the maximum measureable bandwidth of PCI-e.

### Control group: x86-64 with migration resistance
**Parameters**: 32 GiB data, `USE_MANAGED`, `OP_READ`, `ADVISE_PREFERRED_LOCATION_CPU`
```sh
Running on device 0 with grid dim 160 and block dim 128
Managed memory enabled
cudaMemAdviseSetPreferredLocation CPU enabled
Running read kernel
Throughput: 0.530865 GiB/s
Throughput: 10.8151 GiB/s
Throughput: 11.0768 GiB/s
Throughput: 11.4706 GiB/s
Throughput: 11.4655 GiB/s
```
**Observations:** When setting `ADVISE_PREFERRED_LOCATION_CPU` instead of `ADVISE_ACCESSED_BY`, the first run is very slow at only 0.5 GiB/s. After that, we consistently reach the maximum bandwidth again.

### Control group: x86-64 with prefetching
**Example**: `um_prefetch_reprex.cu`

**Parameters**: 32 GiB data, `USE_MANAGED`, `OP_READ`
```sh
Running on device 0 with grid dim 160 and block dim 128
Prefetching 16 MiB data blocks
Managed memory enabled
Running read kernel
Throughput: 8.95776 GiB/s
Throughput: 7.58894 GiB/s
Throughput: 7.58695 GiB/s
Throughput: 7.5479 GiB/s
Throughput: 7.52528 GiB/s
```
**Observations:** Prefetching improves bandwidth over the base settings, but still falls short of the maximum achievable bandwidth (~11 GiB/s).
