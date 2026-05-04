---
jupyter:
  jupytext:
    formats: md,ipynb
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---


# GPU-Accelerated Geant4 with TileCal Geometry and Celeritas

> Checked against upstream documentation in May 2026.
>
> Official Celeritas docs: https://celeritas-project.github.io/celeritas/user/index.html
> Celeritas quick start: https://celeritas-project.github.io/celeritas/user/index.html#quick-start-guide
> Geant4 integration examples: https://github.com/celeritas-project/celeritas/tree/main/example/geant4
> DD4hep project page: https://dd4hep.web.cern.ch/dd4hep/
> Geant4 documentation: https://geant4.web.cern.ch/documentation
> TileCal geometry and macro source: https://github.com/celeritas-project/atlas-tilecal-integration
>
> Note: the TileCal repository is still useful for the GDML geometry and macro cards, but its build and integration instructions predate the current Celeritas Geant4 API. In this notebook we use that repository only as a source of geometry and macro files.

This notebook walks through a minimal, current Geant4 plus Celeritas workflow:

1. Install a recent Celeritas toolchain with Spack.
2. Download the TileCal GDML geometry and macro files.
3. Build a small Geant4 application using the current Celeritas tracking-manager integration path.
4. Compare CPU-only and GPU-enabled runs.
5. Inspect the generated Celeritas diagnostics output.

The example below uses the TileCal geometry distributed in the DD4hep/Celeritas ecosystem, but the executable itself is a plain Geant4 application that loads GDML directly. That keeps the lesson aligned with the current upstream integration examples and avoids stale DDG4-specific code.

---

## 0 Environment setup

### 0.1 Optional Python environment for Jupyter

```bash
# %%bash --no-raise-error
conda create -n tilegpu -y -c conda-forge python=3.11 jupyterlab
conda activate tilegpu
```

### 0.2 Bootstrap Spack

```bash
# %%bash
set -euo pipefail
SPACK_DIR=$HOME/spack
if [ ! -d "$SPACK_DIR" ]; then
  git clone --depth=1 https://github.com/spack/spack.git "$SPACK_DIR"
fi
. "$SPACK_DIR/share/spack/setup-env.sh"
```

Spack is the simplest way to install a consistent Geant4 plus Celeritas stack on Linux or WSL2.

---

## 1 Install Celeritas, Geant4, and DD4hep

The current Celeritas quick-start recommends using Spack for development and integration builds. Geant4 tracking-manager offload requires Geant4 11.0 or newer.

If you have an NVIDIA GPU, set the CUDA architecture that matches your hardware before installing. For example, replace `80` below with the right value for your GPU.

```bash
# %%bash
set -euo pipefail
. "$HOME/spack/share/spack/setup-env.sh"

# Base compiler settings
spack config add packages:all:variants:"cxxstd=17"

# Optional GPU support
spack external find cuda || true
# Uncomment and adjust if you want a GPU-enabled build.
# spack config add packages:all:variants:"+cuda cuda_arch=80"

# Install the packages used in this lesson.
spack install celeritas dd4hep
spack load celeritas dd4hep

# Quick sanity checks
geant4-config --version
ddsim --help >/dev/null 2>&1 || true
spack find --loaded
```

Notes:

- `spack install celeritas` brings in Geant4 and the Geant4-facing Celeritas libraries.
- `dd4hep` is included here so the software stack matches the broader detector-description workflow, even though the minimal example below reads GDML directly.
- If you want to force CPU-only runs later, set `CELER_DISABLE_DEVICE=1` in the runtime environment.

### 1.1 Configure an existing installation for this notebook

If Geant4 and Celeritas are already installed, you do not need to reinstall them for every session. The important part is to start Jupyter from a shell where the Celeritas, Geant4, and DD4hep environment is already loaded.

Recommended launch flow:

```bash
conda activate tilegpu
. "$HOME/spack/share/spack/setup-env.sh"
spack load celeritas dd4hep
jupyter lab
```

Before running the build cells in this notebook, verify that the toolchain is visible:

```bash
. "$HOME/spack/share/spack/setup-env.sh"
spack load celeritas dd4hep
which geant4-config
cmake --version
geant4-config --version
spack find --loaded
```

Two practical notes:

- If you open the notebook from VS Code or Jupyter without launching from a preloaded shell, the Python kernel may not see the same Geant4 and Celeritas environment as your terminal.
- The `%%bash` cells in this notebook explicitly source Spack and load the packages again before building so the notebook does not rely on inherited shell state.

---

## 2 Download the TileCal geometry and macro files

The upstream TileCal repository still provides useful input files. The macro files we use here are `TBrun.mac` and `TBrun_all.mac`.

```python
import pathlib
import urllib.request

tile_repo = "https://raw.githubusercontent.com/celeritas-project/atlas-tilecal-integration/main/"
for filename in (
    "TileTB_2B1EB_nobeamline.gdml",
    "TBrun.mac",
    "TBrun_all.mac",
):
    path = pathlib.Path(filename)
    if not path.exists():
        urllib.request.urlretrieve(tile_repo + filename, path)

print("Downloaded:")
for path in sorted(pathlib.Path(".").glob("TileTB_*")):
    print(" -", path)
for path in sorted(pathlib.Path(".").glob("TBrun*.mac")):
    print(" -", path)
```

---

## 3 Write a minimal Geant4 plus Celeritas application

The current upstream integration pattern is:

1. Include `CeleritasG4.hh`.
2. Use `TrackingManagerIntegration::Instance()`.
3. Register `TrackingManagerConstructor` on your Geant4 physics list.
4. Call `BeginOfRunAction` and `EndOfRunAction` from a Geant4 run action.
5. Link with `Celeritas::G4` and use `celeritas_target_link_libraries(...)` in CMake.

The code below loads the TileCal GDML directly through Geant4's GDML parser and executes a Geant4 macro card.

```cpp
// %%writefile tile_gpu.cc
#include <memory>
#include <string>

#include <FTFP_BERT.hh>
#include <G4GDMLParser.hh>
#include <G4ParticleGun.hh>
#include <G4ParticleTable.hh>
#include <G4RunManagerFactory.hh>
#include <G4SystemOfUnits.hh>
#include <G4ThreeVector.hh>
#include <G4UImanager.hh>
#include <G4UserRunAction.hh>
#include <G4VUserActionInitialization.hh>
#include <G4VUserDetectorConstruction.hh>
#include <G4VUserPrimaryGeneratorAction.hh>

#include <CeleritasG4.hh>

using TMI = celeritas::TrackingManagerIntegration;

namespace
{
class DetectorConstruction final : public G4VUserDetectorConstruction
{
  public:
    G4VPhysicalVolume* Construct() final
    {
        parser_.Read("TileTB_2B1EB_nobeamline.gdml", false);
        return parser_.GetWorldVolume();
    }

  private:
    G4GDMLParser parser_;
};

class PrimaryGeneratorAction final : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction()
        : gun_(1)
    {
        auto* particle
            = G4ParticleTable::GetParticleTable()->FindParticle("e-");
        gun_.SetParticleDefinition(particle);
        gun_.SetParticleEnergy(18 * GeV);
        gun_.SetParticlePosition(G4ThreeVector{0, 0, 0});
        gun_.SetParticleMomentumDirection(G4ThreeVector{1, 0, 0});
    }

    void GeneratePrimaries(G4Event* event) final
    {
        gun_.GeneratePrimaryVertex(event);
    }

  private:
    G4ParticleGun gun_;
};

class RunAction final : public G4UserRunAction
{
  public:
    void BeginOfRunAction(G4Run const* run) final
    {
        TMI::Instance().BeginOfRunAction(run);
    }

    void EndOfRunAction(G4Run const* run) final
    {
        TMI::Instance().EndOfRunAction(run);
    }
};

class ActionInitialization final : public G4VUserActionInitialization
{
  public:
    void BuildForMaster() const final
    {
        this->SetUserAction(new RunAction{});
    }

    void Build() const final
    {
        this->SetUserAction(new PrimaryGeneratorAction{});
        this->SetUserAction(new RunAction{});
    }
};

celeritas::SetupOptions MakeOptions()
{
    celeritas::SetupOptions options;
    options.max_num_tracks = 4096;
    options.initializer_capacity = 4096 * 128;
    options.ignore_processes = {"CoulombScat"};
    options.output_file = "tile_gpu.out.json";
    return options;
}
}  // namespace

int main(int argc, char* argv[])
{
    std::string macro = argc > 1 ? argv[1] : "TBrun.mac";

    std::unique_ptr<G4RunManager> run_manager{
        G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default)};

    run_manager->SetUserInitialization(new DetectorConstruction{});

    auto& tmi = TMI::Instance();
    auto* physics_list = new FTFP_BERT{/* verbosity = */ 0};
    physics_list->RegisterPhysics(
        new celeritas::TrackingManagerConstructor(&tmi));
    run_manager->SetUserInitialization(physics_list);
    run_manager->SetUserInitialization(new ActionInitialization{});

    tmi.SetOptions(MakeOptions());

    run_manager->Initialize();
    G4UImanager::GetUIpointer()->ApplyCommand("/control/execute " + macro);

    return 0;
}
```

Create a matching `CMakeLists.txt` that follows the current upstream example style.

```cmake
# %%writefile CMakeLists.txt
cmake_minimum_required(VERSION 3.18...4.1)
project(tile_gpu LANGUAGES CXX)

find_package(Celeritas 0.6 REQUIRED)
find_package(Geant4 REQUIRED)

if(NOT CELERITAS_USE_Geant4)
  message(FATAL_ERROR "This Celeritas installation was not built with Geant4 support")
endif()

add_executable(tile_gpu tile_gpu.cc)
target_compile_features(tile_gpu PRIVATE cxx_std_17)

celeritas_target_link_libraries(tile_gpu
  Celeritas::G4
  ${Geant4_LIBRARIES}
)
```

Using `0.6` here keeps the example compatible with current stable Celeritas installs while still matching the tracking-manager integration API used in this notebook.

---

## 4 Configure and build

```bash
# %%bash
set -euo pipefail
. "$HOME/spack/share/spack/setup-env.sh"
spack load celeritas dd4hep

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This replaces the older hand-written `g++` command that relied on nonstandard environment variables such as `CELERITAS_INCLUDE_DIRS`. The current upstream recommendation is to use CMake plus `find_package(Celeritas ...)`.

If your Celeritas build includes MPI support and you only want a single-process notebook run, set `CELER_DISABLE_PARALLEL=1` before executing the benchmark cells.

---

## 5 Benchmark CPU-only versus GPU-enabled runs

```python
import os
import pathlib
import subprocess
import time


def run_tile(macro: str, use_gpu: bool):
    env = os.environ.copy()
    env.setdefault("CELER_DISABLE_PARALLEL", "1")
    if not use_gpu:
        env["CELER_DISABLE_DEVICE"] = "1"
    else:
        env.pop("CELER_DISABLE_DEVICE", None)

    exe = pathlib.Path("build/tile_gpu")
    t0 = time.perf_counter()
    result = subprocess.run(
        [str(exe), macro],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    dt = time.perf_counter() - t0
    return dt, result.stdout, result.stderr


cpu_t, cpu_out, cpu_err = run_tile("TBrun.mac", use_gpu=False)
gpu_t, gpu_out, gpu_err = run_tile("TBrun.mac", use_gpu=True)

print(f"CPU wall time: {cpu_t:.2f} s")
print(f"GPU wall time: {gpu_t:.2f} s")
if gpu_t > 0:
    print(f"Speed-up: {cpu_t / gpu_t:.2f}x")
```

Notes:

- `TBrun.mac` currently runs a single 18 GeV `pi+` configuration upstream.
- `CELER_DISABLE_DEVICE=1` is the supported way to force CPU mode for a comparison run.
- The total speed-up depends strongly on how much of the workload is electromagnetic versus hadronic.

---

## 6 Inspect the Celeritas diagnostics output

The example above writes `tile_gpu.out.json`. Its exact contents depend on the Celeritas version and enabled diagnostics, so the safest first step is to inspect the top-level structure rather than hard-code field names.

```python
import json
import pathlib
from pprint import pprint

path = pathlib.Path("tile_gpu.out.json")
if not path.exists():
    raise FileNotFoundError("Run the executable first so tile_gpu.out.json exists")

with path.open() as handle:
    diagnostics = json.load(handle)

print("Top-level keys:")
for key in sorted(diagnostics):
    print(" -", key)

print("\nSelected preview:")
preview = {key: diagnostics[key] for key in list(diagnostics)[:3]}
pprint(preview)
```

If you want a deeper analysis, use this JSON as the starting point for extracting transport counts, kernel timings, or build/runtime configuration details.

---

## 7 Exercises

### Exercise 1: Mixed-particle macro

Run `TBrun_all.mac`, which contains electron, pion, kaon, and proton runs from the upstream TileCal repository.

Questions:

1. Does the overall GPU speed-up decrease compared with `TBrun.mac`?
2. Why is that expected for a workload with a larger hadronic component?

```python
cpu_t, _, _ = run_tile("TBrun_all.mac", use_gpu=False)
gpu_t, _, _ = run_tile("TBrun_all.mac", use_gpu=True)
print(f"TBrun_all.mac speed-up: {cpu_t / gpu_t:.2f}x")
```

### Exercise 2: Event-count scaling

Make a copy of `TBrun.mac`, increase the event count, and see whether the GPU run benefits more from the larger workload.

```python
from pathlib import Path

source = Path("TBrun.mac").read_text()
Path("TBrun_100k.mac").write_text(source.replace("/run/beamOn 10000", "/run/beamOn 100000"))

cpu_t, _, _ = run_tile("TBrun_100k.mac", use_gpu=False)
gpu_t, _, _ = run_tile("TBrun_100k.mac", use_gpu=True)
print(f"TBrun_100k.mac speed-up: {cpu_t / gpu_t:.2f}x")
```

Expected discussion point: GPUs usually amortize setup costs better as the problem size grows, but hadronic-heavy transport can still cap the gain.

---

## Further reading

- Celeritas user documentation: https://celeritas-project.github.io/celeritas/user/index.html
- Celeritas Geant4 examples: https://github.com/celeritas-project/celeritas/tree/main/example/geant4
- DD4hep project documentation: https://dd4hep.web.cern.ch/dd4hep/
- Geant4 documentation portal: https://geant4.web.cern.ch/documentation
- TileCal geometry repository used for this lesson: https://github.com/celeritas-project/atlas-tilecal-integration

---

## 8 Clean-up

```bash
# %%bash --no-raise-error
rm -rf build
rm -f tile_gpu.cc CMakeLists.txt tile_gpu.out.json TBrun.mac TBrun_all.mac TBrun_100k.mac
```
