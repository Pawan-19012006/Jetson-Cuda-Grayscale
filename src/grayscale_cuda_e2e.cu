
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                              \
            std::cerr << "CUDA Error: "                        \
                      << cudaGetErrorString(err) << std::endl; \
            return 1;                                         \
        }                                                     \
    } while (0)

__global__ void grayscaleKernel(
    const unsigned char* input,
    unsigned char* output,
    int totalPixels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < totalPixels) {
        int offset = idx * 3;

        unsigned char b = input[offset];
        unsigned char g = input[offset + 1];
        unsigned char r = input[offset + 2];

        float gray = 0.114f * b + 0.587f * g + 0.299f * r;

        output[idx] = static_cast<unsigned char>(roundf(gray));
    }
}

int main() {

    const std::string imagePath = "data/img2.jpg";

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Could not load image." << std::endl;
        return 1;
    }

    if (!image.isContinuous()) {
        image = image.clone();
    }

    int width = image.cols;
    int height = image.rows;
    int totalPixels = width * height;

    size_t inputBytes = totalPixels * 3 * sizeof(unsigned char);
    size_t outputBytes = totalPixels * sizeof(unsigned char);

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, inputBytes));
    CHECK_CUDA(cudaMalloc(&d_output, outputBytes));

    int threadsPerBlock = 256;
    int blocksPerGrid =
        (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    const int warmupIterations = 5;
    const int iterations = 100;

    // Warm-up
    for (int i = 0; i < warmupIterations; i++) {
        CHECK_CUDA(cudaMemcpy(
            d_input, image.data, inputBytes,
            cudaMemcpyHostToDevice
        ));

        grayscaleKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_input, d_output, totalPixels
        );

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    cv::Mat grayImage(height, width, CV_8UC1);

    double totalH2D = 0.0;
    double totalKernel = 0.0;
    double totalD2H = 0.0;

    using Clock = std::chrono::steady_clock;

    auto pipelineStart = Clock::now();

    for (int i = 0; i < iterations; i++) {

        // 1. Host to Device transfer
        auto h2dStart = Clock::now();

        CHECK_CUDA(cudaMemcpy(
            d_input, image.data, inputBytes,
            cudaMemcpyHostToDevice
        ));

        auto h2dEnd = Clock::now();

        // 2. Kernel execution
        auto kernelStart = Clock::now();

        grayscaleKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_input, d_output, totalPixels
        );

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        auto kernelEnd = Clock::now();

        // 3. Device to Host transfer
        auto d2hStart = Clock::now();

        CHECK_CUDA(cudaMemcpy(
            grayImage.data, d_output, outputBytes,
            cudaMemcpyDeviceToHost
        ));

        auto d2hEnd = Clock::now();

        totalH2D += std::chrono::duration<double, std::milli>(
            h2dEnd - h2dStart
        ).count();

        totalKernel += std::chrono::duration<double, std::milli>(
            kernelEnd - kernelStart
        ).count();

        totalD2H += std::chrono::duration<double, std::milli>(
            d2hEnd - d2hStart
        ).count();
    }

    auto pipelineEnd = Clock::now();

    double totalPipelineMs =
        std::chrono::duration<double, std::milli>(
            pipelineEnd - pipelineStart
        ).count();

    double avgH2D = totalH2D / iterations;
    double avgKernel = totalKernel / iterations;
    double avgD2H = totalD2H / iterations;
    double avgPipeline = totalPipelineMs / iterations;

    // Save final output
    cv::imwrite("results/img2_cuda_e2e_gray.jpg", grayImage);

    std::cout << "\n========== CUDA END-TO-END BENCHMARK ==========\n";

    std::cout << "Image: " << imagePath << "\n";
    std::cout << "Resolution: " << width << " x " << height << "\n";
    std::cout << "Iterations: " << iterations << "\n";

    std::cout << "\nAverage H2D time: "
              << avgH2D << " ms\n";

    std::cout << "Average Kernel time: "
              << avgKernel << " ms\n";

    std::cout << "Average D2H time: "
              << avgD2H << " ms\n";

    std::cout << "Average End-to-End time: "
              << avgPipeline << " ms\n";

    std::cout << "End-to-End FPS: "
              << 1000.0 / avgPipeline << "\n";

    std::cout << "\nOutput saved: results/img2_cuda_e2e_gray.jpg\n";

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
