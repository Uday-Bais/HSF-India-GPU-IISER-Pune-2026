#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <cassert>

namespace Allen {
    enum MemcpyKind {
        memcpyHostToDevice = cudaMemcpyHostToDevice,
        memcpyDeviceToHost = cudaMemcpyDeviceToHost,
        memcpyDeviceToDevice = cudaMemcpyDeviceToDevice
    };

    inline void malloc(void** ptr, size_t size) {
        cudaMalloc(ptr, size);
    }

    inline void memcpy(void* dst, const void* src, size_t size, MemcpyKind kind) {
        cudaMemcpy(dst, src, size, (cudaMemcpyKind)kind);
    }
}

struct MVAModelBase {
    std::string m_name, m_path;
    MVAModelBase(std::string name, std::string path) : m_name(name), m_path(path) {}
    virtual void readData(std::string parameters_path) = 0;
    virtual ~MVAModelBase() = default;
};