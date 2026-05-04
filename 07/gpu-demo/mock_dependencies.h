#pragma once
#include <vector>
#include <string>
#include <cassert>
#include <iostream>
#include <cuda_runtime.h>
#include <map>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

// Mock Allen namespace functions
namespace Allen {
    enum MemcpyKind {
        memcpyHostToDevice = cudaMemcpyHostToDevice,
        memcpyDeviceToHost = cudaMemcpyDeviceToHost,
        memcpyDeviceToDevice = cudaMemcpyDeviceToDevice
    };

    inline void check_cuda(cudaError_t status, const char* operation) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
        }
    }
    
    inline void malloc(void** ptr, size_t size) {
        check_cuda(cudaMalloc(ptr, size), "cudaMalloc");
    }
    
    inline void memcpy(void* dst, const void* src, size_t size, MemcpyKind kind) {
        check_cuda(cudaMemcpy(dst, src, size, (cudaMemcpyKind)kind), "cudaMemcpy");
    }
}

// Mock base class
struct MVAModelBase {
    std::string m_name, m_path;
    MVAModelBase(std::string name, std::string path) : m_name(name), m_path(path) {}
    virtual void readData(std::string parameters_path) = 0;
    virtual ~MVAModelBase() = default;
};
