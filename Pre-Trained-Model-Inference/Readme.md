## How to build and run

### Prerequisites

- CUDA-enabled GPU and NVIDIA drivers (for GPU execution)
- C++ compiler
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (for GPU build)
- [CMake](https://cmake.org/) for building
- Python 3 + Jupyter Notebook (for running notebooks)

### Build the executables

```bash
git clone https://github.com/Uday-Bais/HSF-India-GPU-IISER-Pune-2026.git
cd HSF-India-GPU-IISER-Pune-2026
mkdir build
cd build
cmake ..
make
```

### Run Inference

From the `build` directory:

- **GPU Inference**  
  ```bash
  ./mnist_inference ../network.json ../input.json
  ```

- **CPU Inference**  
  ```bash
  ./mnist_cpu ../network.json ../input.json
  ```

> Replace `../network.json` and `../input.json` with the appropriate paths to your network configuration and input data.


Feedback are welcome!
