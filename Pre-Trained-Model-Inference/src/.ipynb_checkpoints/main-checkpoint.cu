#include "mock_dependencies.h"
#include "demo.cu"
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <fstream>

template <unsigned num_input, unsigned num_hidden1, unsigned num_hidden2, unsigned num_output>
__global__ void mnist_inference_kernel(
    const Allen::MVAModels::DeviceMNISTNetwork<num_input, num_hidden1, num_hidden2, num_output>* model,
    const float* input_data,
    float* output_data,
    int num_tests)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_tests) {
        // Point the model directly at the starting memory address for THIS thread's input.
        // Point the output directly at the starting memory address for THIS thread's output.
        model->evaluate(
            &input_data[idx * num_input], 
            &output_data[idx * num_output]
        );
    }
}

std::vector<std::vector<float>> loadInputsFromJSON(const std::string& json_filepath) {
    std::vector<std::vector<float>> inputs;
    
    nlohmann::json j;
    {
        std::ifstream i(json_filepath);
        if (!i.is_open()) {
            throw std::runtime_error("Cannot open input JSON file: " + json_filepath);
        }
        j = nlohmann::json::parse(i);
    }
    
    inputs = j.at("test_inputs").get<std::vector<std::vector<float>>>();
    
    std::cout << " Loaded " << inputs.size() << " test input(s) from " << json_filepath << std::endl;
    
    return inputs;
}

void run_mnist_inference(const std::string& network_json_path, const std::string& input_json_path) {
    constexpr unsigned num_input = 784;
    constexpr unsigned num_hidden1 = 16;
    constexpr unsigned num_hidden2 = 16;
    constexpr unsigned num_output = 10;

    std::cout << " Network Configuration:\n";
    std::cout << "   Input:     " << num_input << " neurons\n";
    std::cout << "   Hidden 1:  " << num_hidden1 << " neurons (sigmoid)\n";
    std::cout << "   Hidden 2:  " << num_hidden2 << " neurons (sigmoid)\n";
    std::cout << "   Output:    " << num_output << " neurons (sigmoid)\n\n";

    // Parse network path
    std::string net_path, net_filename;
    size_t last_slash_idx = network_json_path.find_last_of("/\\");
    if (std::string::npos != last_slash_idx) {
        net_path = network_json_path.substr(0, last_slash_idx + 1);
        net_filename = network_json_path.substr(last_slash_idx + 1);
    } else {
        net_path = "./";
        net_filename = network_json_path;
    }

    // Load network
    Allen::MVAModels::MNISTNetwork<num_input, num_hidden1, num_hidden2, num_output> model("mnist", net_filename);

    try {
        std::cout << " Loading network from: " << net_path << net_filename << std::endl;
        model.readData(net_path);
        std::cout << " Network loaded successfully!\n\n";
    } catch (const std::exception& e) {
        std::cout << " Error loading network: " << e.what() << "\n";
        return;
    }

    // Load test inputs from input.json
    std::vector<std::vector<float>> test_inputs;
    try {
        std::cout << " Loading test inputs from: " << input_json_path << std::endl;
        test_inputs = loadInputsFromJSON(input_json_path);
        std::cout << "\n";
    } catch (const std::exception& e) {
        std::cout << " Error loading inputs: " << e.what() << "\n";
        return;
    }

    int num_tests = test_inputs.size();
    
    // Prepare host memory
    std::vector<float> host_input(num_tests * num_input);
    std::vector<float> host_output(num_tests * num_output);

    // Flatten inputs into contiguous memory
    for (int i = 0; i < num_tests; i++) {
        if (test_inputs[i].size() != num_input) {
            std::cout << " Input " << i << " has " << test_inputs[i].size() 
                      << " values, expected " << num_input << "\n";
            return;
        }
        for (int j = 0; j < num_input; j++) {
            host_input[i * num_input + j] = test_inputs[i][j];
        }
    }

    // Allocate GPU memory
    float* device_input;
    float* device_output;

    cudaMalloc(&device_input, num_tests * num_input * sizeof(float));
    cudaMalloc(&device_output, num_tests * num_output * sizeof(float));

    // Copy inputs to device
    cudaMemcpy(device_input, host_input.data(), 
               num_tests * num_input * sizeof(float), 
               cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (num_tests + block_size - 1) / block_size;

    auto start = std::chrono::high_resolution_clock::now();

    mnist_inference_kernel<<<grid_size, block_size>>>(
        model.getDevicePointer(), device_input, device_output, num_tests);

    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess) {
        std::cout << " CUDA kernel error: " << cudaGetErrorString(kernel_error) << "\n";
        return;
    }

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Copy results back
    cudaMemcpy(host_output.data(), device_output, 
               num_tests * num_output * sizeof(float), 
               cudaMemcpyDeviceToHost);

    std::cout << " Inference completed in " << duration.count() << " μs\n";

    // Display results for all inputs
    
    std::cout << std::fixed << std::setprecision(4);

    for (int i = 0; i < num_tests; i++) {
        float* probs = &host_output[i * num_output];
        int predicted = std::max_element(probs, probs + num_output) - probs;
        float confidence = *std::max_element(probs, probs + num_output);

        std::cout << "\n   Input " << (i+1) << ":\n";
        std::cout << "    Predicted digit: " << predicted << "\n";
        std::cout << "    Confidence: " << (confidence * 100.0f) << "%\n";
        std::cout << "    Output probabilities:\n";
        for (int j = 0; j < num_output; j++) {
            std::cout << "      [" << j << "]: " << std::setw(8) << probs[j] 
                      << " (" << std::setw(6) << (probs[j] * 100.0f) << "%)\n";
        }
    }

    std::cout << "\n Test completed successfully!\n\n";

    cudaFree(device_input);
    cudaFree(device_output);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_network.json> [path_to_input.json]\n";
        std::cerr << "Example: " << argv[0] << " ./network.json ./input.json\n";
        return 1;
    }

    std::string input_file = (argc >= 3) ? argv[2] : "./input.json";

    int device_count;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        std::cout << " No CUDA devices found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "\nGPU: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";

    run_mnist_inference(argv[1], input_file);

    return 0;
}