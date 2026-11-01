#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>

#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)        \
                      << " at line " << __LINE__ << std::endl;            \
            return false;                                                 \
        }                                                                 \
    } while (0)

// CUDA custom deleters for RAII cleanup on both success and error paths
struct CudaHostDeleter {
    void operator()(unsigned char* ptr) const {
        if (ptr) {
            cudaFreeHost(ptr);
        }
    }
};

struct CudaDeviceDeleter {
    void operator()(unsigned char* ptr) const {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

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

bool fileExists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

bool processImage(const std::string& inputPath, const std::string& outputPath, bool skipIfExists) {
    if (skipIfExists && fileExists(outputPath)) {
        std::cout << "[INFO] Output already exists, preserving: " << outputPath << std::endl;
        return true;
    }

    cv::Mat image = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Could not load image: " << inputPath << std::endl;
        return false;
    }

    if (!image.isContinuous()) {
        image = image.clone();
    }

    int width = image.cols;
    int height = image.rows;
    int totalPixels = width * height;

    size_t inputBytes = totalPixels * 3 * sizeof(unsigned char);
    size_t outputBytes = totalPixels * sizeof(unsigned char);

    // RAII managed pinned host memory
    unsigned char* raw_h_input = nullptr;
    unsigned char* raw_h_output = nullptr;

    CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&raw_h_input), inputBytes, cudaHostAllocDefault));
    std::unique_ptr<unsigned char, CudaHostDeleter> h_input(raw_h_input);

    CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&raw_h_output), outputBytes, cudaHostAllocDefault));
    std::unique_ptr<unsigned char, CudaHostDeleter> h_output(raw_h_output);

    std::memcpy(h_input.get(), image.data, inputBytes);

    // RAII managed device memory
    unsigned char* raw_d_input = nullptr;
    unsigned char* raw_d_output = nullptr;

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&raw_d_input), inputBytes));
    std::unique_ptr<unsigned char, CudaDeviceDeleter> d_input(raw_d_input);

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&raw_d_output), outputBytes));
    std::unique_ptr<unsigned char, CudaDeviceDeleter> d_output(raw_d_output);

    int threadsPerBlock = 256;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    // 1. Host to Device transfer
    CHECK_CUDA(cudaMemcpy(d_input.get(), h_input.get(), inputBytes, cudaMemcpyHostToDevice));

    // 2. Kernel execution
    grayscaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input.get(), d_output.get(), totalPixels);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 3. Device to Host transfer
    CHECK_CUDA(cudaMemcpy(h_output.get(), d_output.get(), outputBytes, cudaMemcpyDeviceToHost));

    // Transfer result to OpenCV Mat and save
    cv::Mat grayImage(height, width, CV_8UC1);
    std::memcpy(grayImage.data, h_output.get(), outputBytes);

    if (!cv::imwrite(outputPath, grayImage)) {
        std::cerr << "Failed to write output image: " << outputPath << std::endl;
        return false;
    }

    std::cout << "[SUCCESS] Processed " << inputPath << " (" << width << " x " << height
              << ") -> " << outputPath << std::endl;

    // RAII smart pointers automatically release all host and device allocations here
    return true;
}

int main(int argc, char** argv) {
    // Single image invocation mode: ./grayscale_cuda_pinned <input_path> <output_path>
    if (argc == 3) {
        return processImage(argv[1], argv[2], false) ? 0 : 1;
    }

    // Default batch mode: process images in project dataset
    struct ImageTask {
        std::string inputPath;
        std::string outputPath;
    };

    std::vector<ImageTask> tasks = {
        {"data/img.jpg", "results/img_cuda_pinned_gray.jpg"},
        {"data/img2.jpg", "results/img2_cuda_pinned_gray.jpg"},
        {"data/img3.jpg", "results/img3_cuda_pinned_gray.jpg"}
    };

    for (const auto& task : tasks) {
        // skipIfExists=true ensures existing outputs (such as img2) are preserved and never overwritten
        if (!processImage(task.inputPath, task.outputPath, true)) {
            return 1;
        }
    }

    return 0;
}
