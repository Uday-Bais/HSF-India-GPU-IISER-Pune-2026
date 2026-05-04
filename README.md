<!--
README Documentation Comment

This README provides an overview of the HSF-India-GPU-IISER-Pune-2026 repository, which contains tutorials, demos, and analysis scripts for GPU programming in the context of the HSF-India parallel programming and GPU course. It details the repository structure, including modules on CUDA basics, unified memory, streaming, profiling, advanced topics, neural network demos, analysis scripts, and notebook verification tools. The README also lists available presentations, describes the contents and purpose of each module, and provides instructions for getting started and using the verification script. Attribution for external code and licensing information are included. The document is intended to guide users through the repository's resources and facilitate learning and analysis of GPU programming techniques.
-->
# HSF-India-GPU-IISER-Pune-2026

This repository provides tutorials, demos, and analysis scripts for GPU programming, tailored for the HSF-India GPU course. Materials are organized into modules covering CUDA basics, unified memory, streaming, profiling, HEP physics usecases and neural network demos.


Presentations:
- Session 1 - https://1drv.ms/p/c/5a70ac10b7f66de0/IQBorVTGUoH2SoGFYU4ZN59QAYw0AdS0aWBhcxr5M6dGjj8?e=KkA54y

- Session 2 - https://1drv.ms/p/c/5a70ac10b7f66de0/IQCIhw6iI0csToQqFloZb6rtAe7Xyb8dd_ALRJRGQrDG78Q?e=Xi59Pa

- Session 3 - https://1drv.ms/p/c/5a70ac10b7f66de0/IQBJghX87dh-SaFOcgFho8kWAf2tTq3QMS_nM1HPWAFLAXM?e=46RNre



## Repository Structure

- **01/** – Introductory CUDA C notebooks and exercises  
  - Presentations (`AC_CUDA_C_*.pptx`)
  - Main tutorial notebook: `Session1.ipynb`
  - Subfolders:  
    - `01-hello/` to `09-heat/`: Step-by-step CUDA examples (hello world, parallelism, indices, loops, memory allocation, error handling, vector addition, matrix multiplication, heat conduction)  
    - `edit/`: Editable files for exercises

- **02/** – Unified Memory tutorials  
  - Main notebook: `Session2_advanced.ipynb`
  - Subfolders: Vector addition, device properties, page faults, prefetching, SAXPY example, and editable files

- **03/** – Streaming and Visual Profiling  
  - Main notebook: `Streaming and Visual Profiling.ipynb`
  - Subfolders: Vector addition, kernel initialization, prefetch checks, stream introduction, manual memory allocation, overlap transfer, n-body simulation, and editable files

- **04/** – Additional CUDA modules  
  - Notebooks: `lesson-5-project.ipynb`, `lesson-5-workbook.ipynb`, `Session3_Python-GPU_HEP.ipynb`

- **05/** – Generator tutorials  
  - Notebooks: `generator_tutorial_gpu.ipynb`, `generator_tutorial_gpu_annotated.ipynb`

- **06/** – HEP analysis and demos  
  - Notebooks: `gpu_dd4hep_tilecal.ipynb`
  - Documentation: `gpu_geant4_dd4hep_cuda_notebook.md`
  - Tutorials: `particle_physics_generators_tutorial.ipynb`

- **07/** – Neural Network Demos and Analysis Scripts  
  - Notebooks: `cuda_neural_network_demo_complete.ipynb`
  - Scripts: `verify_notebook.py`
  - Demos: `gpu-demo/`

- **08/** – Miscellaneous challenges and verification  
  - Code: `01-nbody.cu`
  - Notebook: `GPU challenge.ipynb`



## Notebooks

Each module contains Jupyter notebooks (`.ipynb`) with explanations, code samples, and exercises. Topics include:

- CUDA programming basics
- Memory management (device, unified memory)
- Parallel algorithms (vector/matrix operations, heat conduction)
- Streaming and concurrency
- Profiling and performance analysis
- HEP event generation and analysis
- Neural network demos


## Scripts

- [`07/verify_notebook.py`](07/verify_notebook.py):  
  Analyze notebook structure, extract code cells to files, and report statistics on content and coverage.

## Credits

- The heat conduction CPU source code in `01/AC_CUDA_C.ipynb` is credited to [An OpenACC Example Code for a C-based heat conduction code](http://docplayer.net/30411068-An-openacc-example-code-for-a-c-based-heat-conduction-code.html) from the University of Houston.

## Getting Started

1. Clone the repository.
2. Open the notebooks in Jupyter or VS Code.
3. Follow the step-by-step exercises in each module.
4. Use the verification script to analyze or extract code from notebooks:
     ```sh
     python 08/verify_notebook.py <notebook_path>
     ```

## License

Refer to individual files and notebooks for licensing and attribution information.

---

For more details, explore the subfolders and notebooks in each module.