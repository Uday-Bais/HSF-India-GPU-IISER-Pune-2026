#include "mock_dependencies.h"
#include "MVAModelsManager.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cassert>

namespace Allen::MVAModels {

// ============ DATA STRUCTURES ============

struct MNISTNetworkData {
    std::vector<std::vector<float>> networkw1;
    std::vector<std::vector<float>> networkw2;
    std::vector<std::vector<float>> networkw3;
    std::vector<float> networkb2;
    std::vector<float> networkb3;
    std::vector<float> networkb4;
    
    std::vector<float> fw1;
    std::vector<float> fw2;
    std::vector<float> fw3;
};


// ============ JSON PARSING ============

inline MNISTNetworkData readMNISTNetworkJSON(std::string full_path) {
    MNISTNetworkData to_copy;

    nlohmann::json j;
    {
        std::ifstream i(full_path);
        if (!i.is_open()) {
            throw std::runtime_error("Cannot open JSON file: " + full_path);
        }
        j = nlohmann::json::parse(i);
    }

    using array1d_t = std::vector<float>;
    using array2d_t = std::vector<std::vector<float>>;

    to_copy.networkw1 = j.at("networkw1").get<array2d_t>();
    to_copy.networkw2 = j.at("networkw2").get<array2d_t>();
    to_copy.networkw3 = j.at("networkw3").get<array2d_t>();
    to_copy.networkb2 = j.at("networkb2").get<array1d_t>();
    to_copy.networkb3 = j.at("networkb3").get<array1d_t>();
    to_copy.networkb4 = j.at("networkb4").get<array1d_t>();

    //assert(to_copy.networkw1.size() == 16 && to_copy.networkw1[0].size() == 784);
    //assert(to_copy.networkw2.size() == 16 && to_copy.networkw2[0].size() == 16);
    //assert(to_copy.networkw3.size() == 10 && to_copy.networkw3[0].size() == 16);

    // #updated:
    assert(to_copy.networkw1.size() == 784 && to_copy.networkw1[0].size() == 16);
    assert(to_copy.networkw2.size() == 16 && to_copy.networkw2[0].size() == 16);
    assert(to_copy.networkw3.size() == 16 && to_copy.networkw3[0].size() == 10);
    
    assert(to_copy.networkb2.size() == 16);
    assert(to_copy.networkb3.size() == 16);
    assert(to_copy.networkb4.size() == 10);

    for (const auto& row : to_copy.networkw1) {
        to_copy.fw1.insert(to_copy.fw1.end(), row.begin(), row.end());
    }
    for (const auto& row : to_copy.networkw2) {
        to_copy.fw2.insert(to_copy.fw2.end(), row.begin(), row.end());
    }
    for (const auto& row : to_copy.networkw3) {
        to_copy.fw3.insert(to_copy.fw3.end(), row.begin(), row.end());
    }

    return to_copy;
}


// ============ DEVICE STRUCTURES ============

template <unsigned num_input, unsigned num_hidden1, unsigned num_hidden2, unsigned num_output>
struct DeviceMNISTNetwork {
    constexpr static unsigned nInput = num_input;
    constexpr static unsigned nHidden1 = num_hidden1;
    constexpr static unsigned nHidden2 = num_hidden2;
    constexpr static unsigned nOutput = num_output;

    // FIX 1: Dimensions swapped to match [i][j] access and JSON parsing
    float weights1[nInput][nHidden1];
    float bias2[nHidden1];
    
    float weights2[nHidden1][nHidden2];
    float bias3[nHidden2];
    
    float weights3[nHidden2][nOutput];
    float bias4[nOutput];

    // FIX 2: Added thread_output pointer instead of returning a static array
    __device__ inline void evaluate(const float* input, float* thread_output) const;
};

// ============ HOST CLASS ============

template <unsigned num_input, unsigned num_hidden1, unsigned num_hidden2, unsigned num_output>
struct MNISTNetwork : public MVAModelBase {
    using DeviceType = DeviceMNISTNetwork<num_input, num_hidden1, num_hidden2, num_output>;

    MNISTNetwork(std::string name, std::string path) 
        : MVAModelBase(name, path) {
        m_device_pointer = nullptr;
    }

    const DeviceType* getDevicePointer() const {
        return m_device_pointer;
    }

    void readData(std::string parameters_path) override {
        auto data_to_copy = readMNISTNetworkJSON(parameters_path + m_path);

        Allen::malloc((void**)&m_device_pointer, sizeof(DeviceType));

        constexpr auto size_weights1 = (DeviceType::nHidden1 * DeviceType::nInput) * sizeof(float);
        constexpr auto size_bias2 = DeviceType::nHidden1 * sizeof(float);
        constexpr auto size_weights2 = (DeviceType::nHidden2 * DeviceType::nHidden1) * sizeof(float);
        constexpr auto size_bias3 = DeviceType::nHidden2 * sizeof(float);
        constexpr auto size_weights3 = (DeviceType::nOutput * DeviceType::nHidden2) * sizeof(float);
        constexpr auto size_bias4 = DeviceType::nOutput * sizeof(float);

        Allen::memcpy(m_device_pointer->weights1, data_to_copy.fw1.data(), 
                     size_weights1, Allen::memcpyHostToDevice);
        Allen::memcpy(m_device_pointer->bias2, data_to_copy.networkb2.data(), 
                     size_bias2, Allen::memcpyHostToDevice);

        Allen::memcpy(m_device_pointer->weights2, data_to_copy.fw2.data(), 
                     size_weights2, Allen::memcpyHostToDevice);
        Allen::memcpy(m_device_pointer->bias3, data_to_copy.networkb3.data(), 
                     size_bias3, Allen::memcpyHostToDevice);

        Allen::memcpy(m_device_pointer->weights3, data_to_copy.fw3.data(), 
                     size_weights3, Allen::memcpyHostToDevice);
        Allen::memcpy(m_device_pointer->bias4, data_to_copy.networkb4.data(), 
                     size_bias4, Allen::memcpyHostToDevice);
    }

private:
    DeviceType* m_device_pointer;
};

// ============ ACTIVATION FUNCTIONS ============

namespace ActivateFunction {
    __device__ inline float sigmoid(const float x) {
        return 1.0f / (1.0f + __expf(-x));
    }
}

// ============ FORWARD PASS ============
template <unsigned num_input, unsigned num_hidden1, unsigned num_hidden2, unsigned num_output>
__device__ inline void DeviceMNISTNetwork<num_input, num_hidden1, num_hidden2, num_output>::evaluate(const float* input, float* thread_output) const {
    using ModelType = DeviceMNISTNetwork<num_input, num_hidden1, num_hidden2, num_output>;

    // Layer 1: 784 → 16
    float h1[ModelType::nHidden1] = {0.f};
    #pragma unroll
    for (unsigned j = 0; j < ModelType::nHidden1; j++) {
        #pragma unroll
        for (unsigned i = 0; i < ModelType::nInput; i++) {
            h1[j] += input[i] * weights1[i][j]; 
        }
        h1[j] = ActivateFunction::sigmoid(h1[j] + bias2[j]);
    }

    // Layer 2: 16 → 16
    float h2[ModelType::nHidden2] = {0.f};
    #pragma unroll
    for (unsigned j = 0; j < ModelType::nHidden2; j++) {
        #pragma unroll
        for (unsigned i = 0; i < ModelType::nHidden1; i++) {
            h2[j] += h1[i] * weights2[i][j]; 
        }
        h2[j] = ActivateFunction::sigmoid(h2[j] + bias3[j]);
    }

    // Layer 3: 16 → 10
    #pragma unroll
    for (unsigned j = 0; j < ModelType::nOutput; j++) {
        thread_output[j] = 0.f; // Write directly to the thread-safe output pointer
        #pragma unroll
        for (unsigned i = 0; i < ModelType::nHidden2; i++) {
            thread_output[j] += h2[i] * weights3[i][j]; 
        }
        thread_output[j] = ActivateFunction::sigmoid(thread_output[j] + bias4[j]);
    }
}
}