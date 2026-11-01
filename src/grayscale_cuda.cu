
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

// CUDA kernel: each GPU thread processes one pixel
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

std::string imagePath = "data/img2.jpg";

// Load image using OpenCV
cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

if (image.empty()) {
    std::cerr << "Could not load image: " << imagePath << std::endl;
    return 1;
}

// Ensure image data is contiguous in memory
if (!image.isContinuous()) {
    image = image.clone();
}

int width = image.cols;
int height = image.rows;
int totalPixels = width * height;

size_t inputBytes = totalPixels * 3 * sizeof(unsigned char);
size_t outputBytes = totalPixels * sizeof(unsigned char);

unsigned char* d_input;
unsigned char* d_output;

// Allocate GPU memory
cudaMalloc(&d_input, inputBytes);
cudaMalloc(&d_output, outputBytes);

// Transfer image from CPU to GPU
cudaMemcpy(
    d_input,
    image.data,
    inputBytes,
    cudaMemcpyHostToDevice
);

// Configure CUDA execution
int threadsPerBlock = 256;
int blocksPerGrid =
    (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

// Warm-up
grayscaleKernel<<<blocksPerGrid, threadsPerBlock>>>(
    d_input, d_output, totalPixels
);
cudaDeviceSynchronize();

// Benchmark kernel execution
const int iterations = 100;

auto start = std::chrono::high_resolution_clock::now();

for (int i = 0; i < iterations; i++) {
    grayscaleKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_output, totalPixels
    );
}

cudaDeviceSynchronize();

auto end = std::chrono::high_resolution_clock::now();

double totalTimeMs =
    std::chrono::duration<double, std::milli>(end - start).count();

double avgTimeMs = totalTimeMs / iterations;
double fps = 1000.0 / avgTimeMs;

// Transfer result back to CPU
cv::Mat grayImage(height, width, CV_8UC1);

cudaMemcpy(
    grayImage.data,
    d_output,
    outputBytes,
    cudaMemcpyDeviceToHost
);

// Save output
cv::imwrite("results/img2_cuda_gray.jpg", grayImage);

std::cout << "\nImage: " << imagePath << std::endl;
std::cout << "Resolution: " << width << " x " << height << std::endl;
std::cout << "Average CUDA kernel time: "
            << avgTimeMs << " ms" << std::endl;
std::cout << "CUDA kernel FPS: " << fps << std::endl;
std::cout << "Output saved: results/img2_cuda_gray.jpg" << std::endl;

// Free GPU memory
cudaFree(d_input);
cudaFree(d_output);

return 0;
}
