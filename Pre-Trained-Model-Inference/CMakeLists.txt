cmake_minimum_required(VERSION 3.18)
project(MNISTGPUInference LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Add src directory to include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Also set CUDA include directories explicitly
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/src")

# GPU Inference executable
add_executable(mnist_inference src/main.cu src/demo.cu)

# CPU Inference executable
add_executable(mnist_cpu src/cpu_inference.cu)
target_compile_options(mnist_cpu PRIVATE -std=c++17)

set_target_properties(mnist_inference PROPERTIES
    CUDA_STANDARD 17
    CXX_STANDARD 17
)

set_target_properties(mnist_cpu PROPERTIES
    CXX_STANDARD 17
)
