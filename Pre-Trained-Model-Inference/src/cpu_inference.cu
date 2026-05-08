#include "mock_dependencies.h"
#include "demo.cu"
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <fstream>
#include <cmath>

// ============ CPU DATA STRUCTURES ============

struct CPUMNISTNetworkData {
    std::vector<std::vector<float>> networkw1;  // 784 × 16
    std::vector<std::vector<float>> networkw2;  // 16 × 16
    std::vector<std::vector<float>> networkw3;  // 16 × 10
    std::vector<float> networkb2;                // 16
    std::vector<float> networkb3;                // 16
    std::vector<float> networkb4;                // 10
};

// ============ ACTIVATION FUNCTION (CPU) ============

inline float sigmoid_cpu(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// ============ CPU INFERENCE ============

std::vector<float> cpu_forward_pass(
    const CPUMNISTNetworkData& network,
    const std::vector<float>& input)
{
    const int num_input   = 784;
    const int num_hidden1 = 16;
    const int num_hidden2 = 16;
    const int num_output  = 10;

    // Layer 1: 784 → 16
    std::vector<float> h1(num_hidden1, 0.0f);
    for (int j = 0; j < num_hidden1; j++) {
        for (int i = 0; i < num_input; i++) {
            h1[j] += input[i] * network.networkw1[i][j];
        }
        h1[j] = sigmoid_cpu(h1[j] + network.networkb2[j]);
    }

    // Layer 2: 16 → 16
    std::vector<float> h2(num_hidden2, 0.0f);
    for (int j = 0; j < num_hidden2; j++) {
        for (int i = 0; i < num_hidden1; i++) {
            h2[j] += h1[i] * network.networkw2[i][j];
        }
        h2[j] = sigmoid_cpu(h2[j] + network.networkb3[j]);
    }

    // Layer 3: 16 → 10
    std::vector<float> output(num_output, 0.0f);
    for (int j = 0; j < num_output; j++) {
        for (int i = 0; i < num_hidden2; i++) {
            output[j] += h2[i] * network.networkw3[i][j];
        }
        output[j] = sigmoid_cpu(output[j] + network.networkb4[j]);
    }

    return output;
}

// ============ JSON LOADING ============

CPUMNISTNetworkData loadNetworkJSON_CPU(const std::string& filepath) {
    CPUMNISTNetworkData network;

    nlohmann::json j;
    {
        std::ifstream i(filepath);
        if (!i.is_open()) {
            throw std::runtime_error("Cannot open network JSON file: " + filepath);
        }
        j = nlohmann::json::parse(i);
    }

    using array1d_t = std::vector<float>;
    using array2d_t = std::vector<std::vector<float>>;

    network.networkw1 = j.at("networkw1").get<array2d_t>();
    network.networkw2 = j.at("networkw2").get<array2d_t>();
    network.networkw3 = j.at("networkw3").get<array2d_t>();
    network.networkb2 = j.at("networkb2").get<array1d_t>();
    network.networkb3 = j.at("networkb3").get<array1d_t>();
    network.networkb4 = j.at("networkb4").get<array1d_t>();

    return network;
}

std::vector<std::vector<float>> loadInputsFromJSON_CPU(const std::string& json_filepath) {
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

    return inputs;
}

// ============ MAIN CPU INFERENCE ============

void run_cpu_inference(const std::string& network_json_path, const std::string& input_json_path) {
    constexpr int num_input   = 784;
    constexpr int num_hidden1 = 16;
    constexpr int num_hidden2 = 16;
    constexpr int num_output  = 10;

    // ── TOTAL START ──────────────────────────────────────────────
    auto total_start = std::chrono::high_resolution_clock::now();

    std::cout << " Network Configuration:\n";
    std::cout << "   Input:     " << num_input   << " neurons\n";
    std::cout << "   Hidden 1:  " << num_hidden1 << " neurons (sigmoid)\n";
    std::cout << "   Hidden 2:  " << num_hidden2 << " neurons (sigmoid)\n";
    std::cout << "   Output:    " << num_output  << " neurons (sigmoid)\n\n";

    // Parse network path
    std::string net_path, net_filename;
    size_t last_slash_idx = network_json_path.find_last_of("/\\");
    if (std::string::npos != last_slash_idx) {
        net_path     = network_json_path.substr(0, last_slash_idx + 1);
        net_filename = network_json_path.substr(last_slash_idx + 1);
    } else {
        net_path     = "./";
        net_filename = network_json_path;
    }

    // ── Load network ─────────────────────────────────────────────
    auto load_net_start = std::chrono::high_resolution_clock::now();
    CPUMNISTNetworkData network;
    try {
        network = loadNetworkJSON_CPU(net_path + net_filename);
        std::cout << " Network loaded successfully!\n\n";
    } catch (const std::exception& e) {
        std::cout << "✗ Error loading network: " << e.what() << "\n";
        return;
    }
    auto load_net_end = std::chrono::high_resolution_clock::now();

    // ── Load inputs ──────────────────────────────────────────────
    auto load_input_start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> test_inputs;
    try {
        test_inputs = loadInputsFromJSON_CPU(input_json_path);
        std::cout << " Loaded " << test_inputs.size() << " test input(s)\n\n";
    } catch (const std::exception& e) {
        std::cout << "✗ Error loading inputs: " << e.what() << "\n";
        return;
    }
    auto load_input_end = std::chrono::high_resolution_clock::now();

    int num_tests = static_cast<int>(test_inputs.size());

    // ── Inference ────────────────────────────────────────────────
    auto infer_start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> all_outputs(num_tests);
    for (int i = 0; i < num_tests; i++) {
        if (static_cast<int>(test_inputs[i].size()) != num_input) {
            std::cout << "✗ Input " << i << " has " << test_inputs[i].size()
                      << " values, expected " << num_input << "\n";
            return;
        }
        all_outputs[i] = cpu_forward_pass(network, test_inputs[i]);
    }
    auto infer_end = std::chrono::high_resolution_clock::now();

    // ── Display results ──────────────────────────────────────────
    auto display_start = std::chrono::high_resolution_clock::now();
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < num_tests; i++) {
        float* probs     = all_outputs[i].data();
        int    predicted = std::max_element(probs, probs + num_output) - probs;
        float  confidence = *std::max_element(probs, probs + num_output);

        std::cout << "\n   Input " << (i + 1) << ":\n";
        std::cout << "    Predicted digit: " << predicted << "\n";
        std::cout << "    Confidence:      " << (confidence * 100.0f) << "%\n";
        std::cout << "    Output probabilities:\n";
        for (int j = 0; j < num_output; j++) {
            std::cout << "      [" << j << "]: " << std::setw(8) << probs[j]
                      << " (" << std::setw(6) << (probs[j] * 100.0f) << "%)\n";
        }
    }
    auto display_end = std::chrono::high_resolution_clock::now();

    // ── TOTAL END ────────────────────────────────────────────────
    auto total_end = std::chrono::high_resolution_clock::now();

    // ── Timing summary ───────────────────────────────────────────
    auto us = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };


 
    std::cout << "  TOTAL   time taken      : " << std::setw(10) << us(total_start,      total_end)      << " μs\n";
    std::cout << " ──────────────────────────────────\n";
}

// ============ MAIN ============

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_network.json> [path_to_input.json]\n";
        std::cerr << "Example: " << argv[0] << " ./network.json ./input.json\n";
        return 1;
    }

    std::string input_file = (argc >= 3) ? argv[2] : "./input.json";

    run_cpu_inference(argv[1], input_file);

    return 0;
}