cmake_minimum_required(VERSION 3.21)
project(minimal_optix7_outline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED OptiX_INSTALL_DIR)
  message(FATAL_ERROR "Set -DOptiX_INSTALL_DIR=path/to/NVIDIA-OptiX-SDK when configuring CMake")
endif()

add_executable(minimal_optix7_outline minimal_optix7_outline.cu)
target_include_directories(minimal_optix7_outline PRIVATE ${OptiX_INSTALL_DIR}/include)
target_link_libraries(minimal_optix7_outline PRIVATE CUDA::cudart)
