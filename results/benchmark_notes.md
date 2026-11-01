# Drone Image Preprocessing - Benchmark Notes

## Experiment 001: CPU Grayscale Baseline

**Date:** 2026-11-01

### Hardware

- Device: NVIDIA Jetson Orin Nano Super
- Architecture: ARM64 (aarch64)
- Memory: 8 GB unified memory
- JetPack: 7.2.1
- L4T: r39.2.1
- CUDA Toolkit: 13.2
- OpenCV: 4.8.0

### Experiment

- Operation: BGR to Grayscale conversion
- Implementation: OpenCV CPU (`cv2.cvtColor`)
- Input format: JPEG images loaded as BGR arrays
- Warm-up iterations: 5
- Benchmark iterations: 100
- Timing: `time.perf_counter()`
- Image loading and saving excluded from measured time.

### Results

| Image | Resolution | Average time (ms) | FPS |
|---|---:|---:|---:|
| img2.jpg | 4000 x 2250 | 2.0759 | 481.72 |
| img3.jpg | 4211 x 4078 | 3.6471 | 274.19 |
| img.jpg | 2268 x 4032 | 2.0966 | 476.95 |

### Observations

- The largest image, img3.jpg, has the highest processing time.
- Processing time increases with image resolution.
- CPU measurements represent grayscale conversion only.
- These results serve as the baseline for comparison with custom CUDA implementations.

### Future Experiments

1. Custom CUDA grayscale kernel.
2. CUDA kernel-only execution time.
3. CUDA end-to-end execution time, including host-device transfers.
4. CPU versus CUDA comparison.
5. Output correctness comparison.
6. Performance profiling and optimization.

## Experiment 002: Custom CUDA Grayscale Kernel

**Date:** 2026-11-01

### Implementation
- Language: CUDA C++
- Compiler: NVCC 13.2
- Image loading/saving: OpenCV 4.8.0
- Input: img2.jpg (4000 x 2250)
- Total pixels: 9,000,000
- Threads per block: 256
- Blocks: 35,157
- Warm-up iterations: 1
- Benchmark iterations: 100

### Results

| Metric | Result |
|---|---:|
| Average kernel time | 1.19187 ms |
| Kernel throughput | 839.016 FPS |

### Initial CPU Comparison

| Implementation | Average time |
|---|---:|
| CPU OpenCV | 2.0759 ms |
| Custom CUDA kernel | 1.19187 ms |

Initial kernel-only speedup: approximately 1.74x.

**Important:** CUDA timing excludes host-device memory transfers. This is not an end-to-end comparison.

### Next Steps
1. Verify CUDA output against CPU output.
2. Measure CUDA host-to-device transfer time.
3. Measure CUDA device-to-host transfer time.
4. Measure complete CUDA processing time.
5. Benchmark all three input images.

## Experiment 003: CPU vs CUDA Output Validation

**Date:** 2026-11-01
**Image:** img2.jpg

### Validation Method
- Compared decoded CPU and CUDA grayscale JPEG outputs.
- Used OpenCV `cv2.absdiff()`.
- Measured mean and maximum absolute pixel differences.
- Accepted tolerance: 2 intensity levels.

### Results

| Metric | Result |
|---|---:|
| CPU dimensions | 4000 x 2250 |
| CUDA dimensions | 4000 x 2250 |
| Mean absolute difference | 0.00000667 |
| Maximum absolute difference | 1 |
| Pixels within tolerance 2 | 100% |

### Conclusion
CUDA output matches CPU output with a maximum pixel difference of 1.
The implementation passes the current output correctness validation.

## Experiment 004: CUDA End-to-End Benchmark

**Date:** 2026-11-01

### Configuration
- Device: Jetson Orin Nano Super
- Image: img2.jpg
- Resolution: 4000 x 2250
- Total pixels: 9,000,000
- Threads per block: 256
- Warm-up iterations: 5
- Benchmark iterations: 100

### Results

| Stage | Average latency |
|---|---:|
| Host to Device | 4.13267 ms |
| CUDA kernel (host timed with synchronization) | 1.55612 ms |
| Device to Host | 1.55061 ms |
| Complete pipeline | 7.24153 ms |

End-to-end throughput: 138.092 FPS.

### CPU Comparison

- CPU OpenCV: 2.0759 ms
- CUDA kernel-only (previous experiment): 1.19187 ms
- CUDA end-to-end: 7.24153 ms

The complete CUDA implementation is currently slower than CPU OpenCV for this operation.

### Observations
- Host-to-device transfer is the largest individual stage.
- Data transfer overhead outweighs the kernel execution advantage.
- Kernel-only and synchronized host-timed measurements use different methodologies and must be reported separately.
- End-to-end timing includes the complete transfer and processing loop.

### Optimization Opportunities
1. Investigate pinned host memory.
2. Investigate asynchronous memory transfers.
3. Explore CUDA streams.
4. Keep frames on the GPU across multiple CV operations.
5. Avoid unnecessary GPU-to-CPU transfers.

## Experiment 005: End-to-End Output Validation

**Date:** 2026-11-01
**Image:** img2.jpg

### Results

| Metric | Result |
|---|---:|
| Mean absolute difference | 0.00000667 |
| Maximum absolute difference | 1 |
| Pixels within tolerance 2 | 100% |

### Conclusion
The end-to-end CUDA grayscale implementation passes output correctness validation against CPU OpenCV.
Maximum observed pixel difference is 1 intensity level.

## Experiment 006: Pinned Host Memory

**Date:** 2026-11-01
**Image:** img2.jpg (4000 x 2250)
**Warmup:** 5 iterations
**Benchmark:** 100 iterations

### Implementation
- Allocated host buffers using cudaHostAlloc().
- Used pinned memory for both input and output.
- Used synchronous cudaMemcpy() for transfers.
- CUDA kernel: 256 threads per block.
- Pinned buffer initialization excluded from timed iterations.

### Results

| Metric | Average |
|---|---:|
| H2D | 1.16549 ms |
| Kernel | 1.55175 ms |
| D2H | 0.42269 ms |
| End-to-End | 3.14117 ms |
| FPS | 318.353 |

### Comparison with pageable-memory CUDA

| Metric | Pageable | Pinned |
|---|---:|---:|
| H2D | 4.13267 ms | 1.16549 ms |
| Kernel | 1.55612 ms | 1.55175 ms |
| D2H | 1.55061 ms | 0.42269 ms |
| End-to-End | 7.24153 ms | 3.14117 ms |

End-to-end latency reduction: approximately 56.62%.
End-to-end speedup: approximately 2.31x.

### Observation
Pinned host memory substantially reduced measured transfer times.
Kernel execution time remained approximately unchanged.

### Limitation
The OpenCV-to-pinned-host-memory copy and pinned-memory allocation are outside the timed pipeline. Results represent repeated processing with an already-prepared pinned input buffer.
