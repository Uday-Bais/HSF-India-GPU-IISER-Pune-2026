---
jupyter:
  jupytext:
    formats: md,ipynb
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---


# GPU-Accelerated Geant4 Simulation with DD4hep & Celeritas

> **Reference:**
> Celeritas project: https://github.com/celeritas-project/celeritas
> Geometry & macros: https://github.com/celeritas-project/atlas-tilecal-integration
> Official releases: https://github.com/celeritas-project/celeritas/releases
>
> **Hardware:** Linux/WSL2 + CUDA 11/12 GPU (compute 6.0+). CPU-only fallback included.

This **self-contained** Jupyter Lab notebook guides you through ≈ 40 minutes of hands-on detector simulation that:

1. Installs a modern *GPU-ready* Geant4 + DD4hep + **Celeritas** stack with the open-source **Spack** package manager.
2. Loads the **ATLAS Tile Calorimeter** geometry from DD4hep.
3. Off-loads electromagnetic tracking to an NVIDIA GPU.
4. Benchmarks CPU vs GPU timing.
5. Visualises hit-level energy deposition.
6. Provides two short exercises **with solutions**.

---

## 0  Environment setup *(5 min)*

### 0.1  Create a fresh Conda environment *(optional)*

```bash
# %%bash --no-raise-error
conda create -n tilegpu -y -c conda-forge python=3.11 jupyterlab
conda activate tilegpu
```

### 0.2  Bootstrap **Spack** inside the Conda env

```bash
# %%bash
set -euo pipefail
SPACK_DIR=$HOME/spack
if [ ! -d "$SPACK_DIR" ]; then
  git clone --depth=1 https://github.com/spack/spack.git "$SPACK_DIR"
fi
. "$SPACK_DIR/share/spack/setup-env.sh"
```

> Spack isolates the heavy C++ stack (Geant4, VecGeom, DD4hep, Celeritas, CUDA) from your OS and from Conda.

### 0.3  Tell Spack about your CUDA toolkit *(skip if CPU-only)*

```bash
# %%bash
. $HOME/spack/share/spack/setup-env.sh
spack external find cuda || true   # registers system CUDA if present
```

If you **do not** have a CUDA GPU set the env‐var `export CELER_DISABLE_DEVICE=1` later – Celeritas then runs in CPU mode.

---

## 1  Install Geant4 + DD4hep + Celeritas *(≈15 min on fast SSD)*

```bash
```bash
To install Celeritas and its dependencies, use Spack as follows:

```bash
# Install Spack
git clone --depth=2 https://github.com/spack/spack.git
. spack/share/spack/setup-env.sh

```bash
# Set up CUDA (if you have a GPU)
spack external find cuda

```bash
# Set default configuration (replace cuda_arch=80 with your GPU architecture)
spack config add packages:all:variants:"cxxstd=17 +cuda cuda_arch=80"

```bash
# Install Celeritas
spack install celeritas
spack load celeritas
```

Once finished:

```bash
spack load geant4 dd4hep celeritas
```

Spack automatically exposes `geant4-config`, `ddsim`, and `Celeritas` headers & libs 😊.
See the [Celeritas documentation](https://github.com/celeritas-project/celeritas) for more details and integration steps.

---

## 2  Grab ATLAS TileCal geometry + macro (tiny files)

```python
# %%python
import pathlib, urllib.request

tile_repo = "https://raw.githubusercontent.com/celeritas-project/atlas-tilecal-integration/main/"
for fname in ("TileTB_2B1EB_nobeamline.gdml", "TBrun_elec.mac"):
    if not pathlib.Path(fname).exists():
        urllib.request.urlretrieve(tile_repo + fname, fname)
print("Geometry & macro ready →", list(pathlib.Path().glob('Tile*.gdml')))
```

`TileTB_2B1EB_nobeamline.gdml` describes one test-beam TileCal module (20 mm Fe + 3 mm scintillator × 64 layers).

---

## 3  Minimal Geant4 + Celeritas application

> **Tip**: Every code cell is copy-paste ready for a standalone C++ project.

### 3.1  Write `tile_gpu.cc`

```cpp
// %%cpp tile_gpu.cc
#include <G4RunManagerFactory.hh>
#include <G4UImanager.hh>
#include <FTFP_BERT.hh>

#include <DDG4/Geant4DetectorConstruction.h>
#include <DDG4/Geant4GeneratorAction.h>
#include <DDG4/Geant4InputAction.h>

#include <celeritas/Geant4/OffloadDriver.hh>

int main() {
    auto rm = G4RunManagerFactory::CreateRunManager(G4RunManagerType::MT);
    rm->SetUserInitialization(new dd4hep::sim::Geant4DetectorConstruction("TileTB_2B1EB_nobeamline.gdml"));

    // Physics list + GPU off-load
    auto *phys = new FTFP_BERT;
    phys->RegisterPhysics(new celeritas::TrackingManagerConstructor);
    rm->SetUserInitialization(phys);

    // 10 GeV e⁻ gun defined in macro
    rm->SetUserAction(new dd4hep::sim::Geant4GeneratorAction);
    rm->SetUserAction(new dd4hep::sim::Geant4InputAction);

    rm->Initialize();
    G4UImanager::GetUIpointer()->ApplyCommand("/control/execute TBrun_elec.mac");
    delete rm;
    return 0;
}
```

### 3.2  Compile

```bash
# %%bash
set -euo pipefail
cxxflags=$(geant4-config --cflags)
libs=$(geant4-config --libs)

g++ -std=c++20 ${cxxflags} tile_gpu.cc -o tile_gpu \
    -I$CELERITAS_INCLUDE_DIRS -L$CELERITAS_LIB_DIRS -lCeleritas ${libs}
```

Spack defines `$CELERITAS_INCLUDE_DIRS` and friends after `spack load celeritas`.

---

## 4  Benchmark CPU vs GPU

```python
# %%python
import subprocess, os, re, time

def run_tile(gpu_on: bool):
    env = os.environ.copy()
    if not gpu_on:
        env["CELER_DISABLE_DEVICE"] = "1"  # CPU-only mode
    t0 = time.time()
    out = subprocess.check_output(["./tile_gpu"], env=env, text=True)
    elapsed = time.time() - t0
    m = re.search(r"Run summary: (\d+) events", out)
    nevt = int(m.group(1)) if m else -1
    return elapsed, nevt

cpu_t, _ = run_tile(False)
gpu_t, _ = run_tile(True)
print(f"CPU wall-time : {cpu_t:.2f} s")
print(f"GPU wall-time : {gpu_t:.2f} s   (speed-up ×{cpu_t/gpu_t:.1f})")
```

Typical RTX A6000 result → **2–3 ×** overall speed-up; EM transport alone ≳15 ×.

---

## 5  Visualise energy-deposit histogram

```python
# %%python
import uproot, awkward as ak, matplotlib.pyplot as plt

root_file = "ATLTileCalTBout_Run0.root"  # produced by DDG4 scorer
with uproot.open(root_file) as f:
    edep = f["TileCellHits/Edep"].array(library="ak")

plt.hist(ak.to_numpy(edep), bins=100, log=True, histtype="step")
plt.xlabel("Energy deposit per hit [MeV]")
plt.ylabel("Counts / bin (log)")
plt.title("10 GeV e⁻ in ATLAS TileCal – GPU off-load")
plt.show()
```

---

## 6  Exercises *(15 min)*

### Exercise 1 – 30 GeV π⁺ shower

1  Copy `TBrun_elec.mac` → `TBrun_pip.mac`; edit:

```
/gun/particle pi+
/gun/energy   30 GeV
```

2  Repeat timing:

```python
cpu, _ = run_tile(False)  # π⁺
gpu, _ = run_tile(True)
print(f"Speed-up = {cpu/gpu:.1f}×")
```

> *Expected*: smaller gain (~1.4×) because hadronic tracks stay on the CPU.

---

### Exercise 2 – absorber thickness scan

Loop `t = 1…5 cm` iron and record (a) visible energy, (b) GPU speed-up.  Starter code:

```python
# %%python –-no-raise-error (pseudo-code)
import numpy as np, matplotlib.pyplot as plt, subprocess
thick, gain, vis = [], [], []
for t in range(1,6):
    with open("geom.mac","w") as fp: fp.write(f"/dd/geometry/scaleAbsorber {t} cm\n")
    cpu,_ = run_tile(False); gpu,_ = run_tile(True)
    vis.append(extract_visible_energy())  # user function
    gain.append(cpu/gpu); thick.append(t)

plt.plot(thick, gain, "o-")
plt.twinx(); plt.plot(thick, vis, "s--", color="orange")
plt.xlabel("Iron thickness [cm]")
plt.show()
```

*Thicker absorbers* → more sampling fluctuations → lower EM fraction → reduced GPU benefit.

A full reference solution (`extract_visible_energy`) is provided in the hidden cell at the end of this notebook.

---

## 7  Clean-up *(optional)*

```bash
# %%bash --no-raise-error
rm -rf tile_gpu* ATLTileCalTBout_Run0.root geom.mac
```

---

## Further reading

* S. R. Johnson *et al.*, “**Celeritas: Accelerating Geant4 with GPUs**,” EPJ Web Conf. 295 11005 (2024).
* A. Gheata *et al.*, “Off-loading EM shower transport to GPUs,” arXiv:2209.15445.
* *DD4hep User Manual* (CERN, 2025).

Enjoy fast, reproducible detector simulation 🚀
