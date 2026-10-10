# Cardiac Neural Operators

Code for **Neural Operators for Cardiac Field Reconstruction from Simulated Magnetic Sensors**. FNO and DeepONet reconstruct activation and repolarization fields from simulated magnetic measurements on a fixed 2D tissue sheet.

The package contains simulation, training, evaluation and control scripts, plus compact arrays and tables for checking the additional controls. Full simulation datasets, trained checkpoints and historical training logs are not included.

## Verify the saved controls

With Python 3.12:

```bash
python -m pip install numpy pandas
python pipeline.py verify
```

This recomputes the training-mean and fixed-coordinate control scores from the maps in `supplemental/`.

## Run the benchmark

Use Linux, openCARP, carputils and a CUDA-capable PyTorch installation. Set `OPENCARP_MESH_PREFIX` if the reference mesh is outside the standard openCARP installation.

```bash
python -m pip install -r requirements.txt
python pipeline.py all --dry-run
python pipeline.py --help
```

Remove `--dry-run` to execute the full sequence, including 5,000 simulations and model training. Individual stages and their options are listed by `--help`. Existing evaluation outputs are protected against overwrite. Historical retraining can produce different validation selections; see the CLI help for reproduction constraints.

`scripts/` contains the implementation; `FILE_MANIFEST.csv` records file checksums. The paper describes the experimental settings and limitations.

This is a synthetic research benchmark, without validation on patient anatomy or measured clinical data. Code: MIT (`LICENSE`). Included result arrays and tables: CC BY 4.0. External software licenses are listed in `THIRD_PARTY_LICENSES.txt`.
