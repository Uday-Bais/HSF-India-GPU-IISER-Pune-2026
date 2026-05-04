Session 04 data files

Place the external data files needed by the Session 04 notebooks in this directory.

Currently present in this checkout:
- `SMHiggsToZZTo4L.root`
  Source: https://opendata.cern.ch/record/12361
  Verification: matches the notebook expectation `SMHiggsToZZTo4L.root:Events` and contains the required `Electron_pt`, `Electron_eta`, `Electron_phi`, and `Electron_charge` branches.
- `Mudemo.root`
  Source: https://opendata.cern.ch/record/5001
  Verification: this is not the file expected by the current Session 04 notebooks. It contains histogram objects under `demo/` and does not provide the project notebook input `dimuon_mass.root:tree/mass`.

Confirmed public source:
- `SMHiggsToZZTo4L.root`: CERN Open Data record 12361
  https://opendata.cern.ch/record/12361
- `Mudemo.root`: CERN Open Data record 5001
  https://opendata.cern.ch/record/5001

Expected by the local Session 04 notebooks:
- `SMHiggsToZZTo4L.root`
- `dimuon_mass.root`

Current compatibility status:
- `SMHiggsToZZTo4L.root`: verified working for `Session3_Python-GPU_HEP.ipynb`, `lesson-4-workbook.ipynb`, and the ROOT-only rewrite of Exercise 1 in `lesson-4-project.ipynb`.
- `dimuon_mass.root`: still missing.
- `Mudemo.root`: present, but not a compatible replacement for `dimuon_mass.root`.

Related upstream course formats used in other lessons:
- `SMHiggsToZZTo4L.json`
- `SMHiggsToZZTo4L.parquet`

The Session 04 notebooks read files from `04/data/` directly. They do not generate these HEP datasets in notebook cells.