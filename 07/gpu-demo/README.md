# CUDA Neural Network Demo

A CUDA-accelerated single-layer fully connected neural network implementation for GPU evaluation.

## Features

- **GPU-accelerated inference**: Neural network evaluation runs entirely on GPU using CUDA
- **Template-based design**: Compile-time configuration of input size and hidden nodes
- **Optimized CUDA kernels**: Uses loop unrolling and fast math operations for performance
- **Random model generation**: Utility to generate random weights and biases for testing
- **Comprehensive testing**: Includes validation and statistics

## Neural Network Architecture

- **Input Layer**: Configurable size (template parameter)
- **Hidden Layer**: Single fully connected layer with ReLU activation
- **Output Layer**: Single neuron with sigmoid activation (0-1 output range)
- **Data preprocessing**: Input normalization using mean and standard deviation

## Files

- `demo.cu` - Main neural network implementation
- `main.cu` - Test program with CUDA kernel
- `nn_gen/json_generator.cpp` - Utility to generate random model parameters
- `mock_dependencies.h` - Mock implementations for Allen framework dependencies
- `CMakeLists.txt` - Build system configuration for CMake
- Generated JSON model files such as `test_model.json` - Created by `json_generator` during the workflow

## Compilation

### Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (tested with CUDA 12.x)
- C++17 compatible compiler
- CMake (3.18+)
- nlohmann-json library (system-wide installation)

All commands below assume you are running them from the `07/gpu-demo` directory.

### Build Commands

```bash
# Configure the project
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Optional example for a Turing GPU
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75

# Compile the executables
cmake --build build --config Release
```

On Linux or WSL with Makefiles or Ninja, this will usually create `json_generator` and `neural_network_test` under `build/`.
On Windows with a multi-config generator such as Visual Studio, they are typically under `build/Release/`.

### GPU Architecture Support

The current `CMakeLists.txt` does not hard-code a single CUDA architecture. If you want to target a specific GPU family explicitly, pass `CMAKE_CUDA_ARCHITECTURES` during configuration.

- `60` or `61`: Pascal / GTX 10 series
- `70`: Volta / Titan V
- `75`: Turing / RTX 20 series and GTX 16 series
- `86`: Ampere / RTX 30 series
- `89`: Ada / RTX 40 series

## Usage

Most usage examples below assume you are running them from the `build` directory on Linux or WSL.

### 1. Generate Random Model

```bash
./json_generator [num_input] [num_node] [output_file]
```

Example:

```bash
./json_generator 4 8 ../my_model.json
```

If you use Visual Studio on Windows, run the executable from `build/Release/` instead.

### 2. Run Neural Network Test

```bash
./neural_network_test [path_to_model.json]
```

Example:

```bash
./neural_network_test ../my_model.json
```

### JSON Model Format

```json
{
  "num_input": 4,
  "num_node": 8,
  "mean": [0.5, 1.2, -0.3, 2.1],
  "std": [1.0, 0.8, 1.5, 0.9],
  "weights1": [
    [0.234, -0.567, 0.891, 0.123],
    [-0.456, 0.789, 0.345, -0.678],
    ...
  ],
  "bias1": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
  "weights2": [0.8, -0.6, 0.4, -0.2, 0.9, -0.7, 0.5, -0.3],
  "bias2": 0.15
}
```

## Test Output

The test program will:

1. Load the neural network model from the specified JSON file.
2. Generate random test inputs.
3. Launch a CUDA kernel for parallel evaluation.
4. Display results and statistics.
5. Validate that outputs are in the expected range [0, 1].

Example output:

```text
=== Test Results ===
Test  1: Input [-0.527618, -0.339161, 1.488258, -0.412800] -> Output: 0.947760
Test  2: Input [0.694293, 1.286170, 1.175678, -0.213452] -> Output: 0.694606
...
=== Validation ===
✓ All outputs are in valid range [0, 1] (sigmoid activation)
Output statistics:
- Min: 0.382523
- Max: 0.952799
- Average: 0.793204
```

## Performance Notes

- Uses CUDA's `__device__` functions for GPU execution
- Loop unrolling with `#pragma unroll` for better performance
- Template-based design allows compile-time optimizations
- Supports concurrent evaluation of multiple inputs

## Original Framework

This code is adapted from the Allen framework (CERN LHCb Collaboration) with mock dependencies for standalone compilation and testing.

## License

Apache License 2.0 (inherited from original Allen framework)
