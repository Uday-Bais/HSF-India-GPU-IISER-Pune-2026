cmake_minimum_required(VERSION 3.18)
project(GpuDemoNN LANGUAGES CXX CUDA)

find_package(nlohmann_json 3.2.0 REQUIRED)

add_executable(neural_network_test main.cu demo.cu)
target_link_libraries(neural_network_test PRIVATE nlohmann_json::nlohmann_json)
set_target_properties(neural_network_test PROPERTIES
    CUDA_STANDARD 17
    CXX_STANDARD 17
)

add_executable(json_generator nn_gen/json_generator.cpp)
target_link_libraries(json_generator PRIVATE nlohmann_json::nlohmann_json)
set_target_properties(json_generator PROPERTIES
    CXX_STANDARD 17
)
