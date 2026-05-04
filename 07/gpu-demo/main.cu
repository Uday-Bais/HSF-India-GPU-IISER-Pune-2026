#include "mock_dependencies.h"
#include "demo.cu"
#include <iostream>
#include <random>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <stdexcept>

namespace {
void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}
}

// CUDA kernel to test neural network evaluation
template<unsigned num_input, unsigned num_node>
__global__ void test_neural_network_kernel(
    const Allen::MVAModels::DeviceSingleLayerFCNN<num_input, num_node>* model,
    float* input_data,
    float* output_data,
    int num_tests)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_tests) {
        // Each thread processes one test case
        float local_input[num_input];
        for (int i = 0; i < num_input; i++) {
            local_input[i] = input_data[idx * num_input + i];
        }
        
        // Evaluate the neural network
        output_data[idx] = model->evaluate(local_input);
    }
}

// Host function to test neural network evaluation
void test_neural_network_evaluation(const std::string& json_filepath) {
    constexpr unsigned num_input = 4;
    constexpr unsigned num_node = 8;
    constexpr int num_tests = 10;
    
    std::cout << "\n=== Neural Network Evaluation Test ===" << std::endl;
    std::cout << "Network configuration:" << std::endl;
    std::cout << "- Input size: " << num_input << std::endl;
    std::cout << "- Hidden nodes: " << num_node << std::endl;
    std::cout << "- Number of test cases: " << num_tests << std::endl;
    
    // Resolve model paths in a platform-safe way.
    const std::filesystem::path model_path {json_filepath};
    if (!std::filesystem::exists(model_path)) {
        std::cout << "✗ Model file not found: " << model_path << std::endl;
        return;
    }

    const auto parent = model_path.parent_path();
    const std::string path = parent.empty() ? "./" : (parent.string() + std::filesystem::path::preferred_separator);
    const std::string filename = model_path.filename().string();

    Allen::MVAModels::SingleLayerFCNN<num_input, num_node> model("test_model", filename);
    
    try {
        // Load model data (this will use our mock implementation)
        model.readData(path);
        std::cout << "✓ Model loaded successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Error loading model: " << e.what() << std::endl;
        return;
    }
    
    // Generate random test input data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
    
    std::vector<float> host_input(num_tests * num_input);
    std::vector<float> host_output(num_tests);
    
    std::cout << "\nGenerating random test inputs..." << std::endl;
    for (int i = 0; i < num_tests * num_input; i++) {
        host_input[i] = dis(gen);
    }
    
    // Allocate device memory
    float* device_input = nullptr;
    float* device_output = nullptr;

    try {
        check_cuda(cudaMalloc(&device_input, num_tests * num_input * sizeof(float)), "cudaMalloc(device_input)");
        check_cuda(cudaMalloc(&device_output, num_tests * sizeof(float)), "cudaMalloc(device_output)");

        // Copy input data to device
        check_cuda(
            cudaMemcpy(
                device_input,
                host_input.data(),
                num_tests * num_input * sizeof(float),
                cudaMemcpyHostToDevice),
            "cudaMemcpy(host_input -> device_input)");

        // Launch kernel
        int block_size = 256;
        int grid_size = (num_tests + block_size - 1) / block_size;

        std::cout << "Launching CUDA kernel..." << std::endl;
        std::cout << "Grid size: " << grid_size << ", Block size: " << block_size << std::endl;

        test_neural_network_kernel<<<grid_size, block_size>>>(
            model.getDevicePointer(), device_input, device_output, num_tests);

        check_cuda(cudaGetLastError(), "kernel launch");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        // Copy results back to host
        check_cuda(
            cudaMemcpy(
                host_output.data(),
                device_output,
                num_tests * sizeof(float),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(device_output -> host_output)");
    }
    catch (const std::exception& e) {
        if (device_input) cudaFree(device_input);
        if (device_output) cudaFree(device_output);
        std::cout << "✗ CUDA runtime error: " << e.what() << std::endl;
        return;
    }
    
    // Display results
    std::cout << "\n=== Test Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    
    for (int i = 0; i < num_tests; i++) {
        std::cout << "Test " << std::setw(2) << (i + 1) << ": ";
        std::cout << "Input [";
        for (int j = 0; j < num_input; j++) {
            std::cout << std::setw(8) << host_input[i * num_input + j];
            if (j < num_input - 1) std::cout << ", ";
        }
        std::cout << "] -> Output: " << std::setw(8) << host_output[i] << std::endl;
    }
    
    // Verify outputs are in expected range (sigmoid output should be 0-1)
    bool all_valid = true;
    for (int i = 0; i < num_tests; i++) {
        if (host_output[i] < 0.0f || host_output[i] > 1.0f) {
            all_valid = false;
            break;
        }
    }
    
    std::cout << "\n=== Validation ===" << std::endl;
    if (all_valid) {
        std::cout << "✓ All outputs are in valid range [0, 1] (sigmoid activation)" << std::endl;
    } else {
        std::cout << "✗ Some outputs are outside valid range [0, 1]" << std::endl;
    }
    
    // Calculate some basic statistics
    float min_output = *std::min_element(host_output.begin(), host_output.end());
    float max_output = *std::max_element(host_output.begin(), host_output.end());
    float avg_output = std::accumulate(host_output.begin(), host_output.end(), 0.0f) / num_tests;
    
    std::cout << "Output statistics:" << std::endl;
    std::cout << "- Min: " << min_output << std::endl;
    std::cout << "- Max: " << max_output << std::endl;
    std::cout << "- Average: " << avg_output << std::endl;
    
    // Clean up
    cudaFree(device_input);
    cudaFree(device_output);
    
    std::cout << "\n✓ Test completed successfully!" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "CUDA Neural Network Demo" << std::endl;
    std::cout << "========================" << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model.json>" << std::endl;
        return 1;
    }
    
    // Check CUDA device
    int device_count = 0;
    cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess) {
        std::cout << "✗ Failed to query CUDA devices: " << cudaGetErrorString(device_status) << std::endl;
        return 1;
    }
    
    if (device_count == 0) {
        std::cout << "✗ No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaError_t prop_status = cudaGetDeviceProperties(&prop, 0);
    if (prop_status != cudaSuccess) {
        std::cout << "✗ Failed to query CUDA device properties: " << cudaGetErrorString(prop_status) << std::endl;
        return 1;
    }
    std::cout << "Using CUDA device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    
    // Run the neural network test
    test_neural_network_evaluation(argv[1]);
    
    return 0;
}
