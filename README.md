## Repository Structure

```text
msc-thesis-code/
├── README.md
│
├── data_generation/                        # Data-generating processes and simulations
│   ├── barabasi_albert_datagen.py        
│   ├── binomial_edgeweight_datagen.py          
│   └── topology_covariance_datagen.py                  
│
├── methods/                                # Classifier and statistical testing methods
│   ├── ginestet2017.py
│   ├── dubey2019.py
│   ├── lovato2020.py
│   ├── knn.py
│   ├── kernel_svm.py
│   └── gcn.py
│
├── testing/                                
│   ├── permutation_test.py                 
│
├── simulations/                            # Pipeline execution and power curve scripts
│   ├── ba_sweep.py        
│   ├── ba_sweep_resubstitution_n25.py                 
│   ├── binomial_edgeweight_sweep.py                
│   └── topology_covariance_sweep.py 
│
├── diagnostics/                            # Diagnostic and ablation studies
│   ├── ba_distance_asymmetry_check.py         # Mean pairwise Frobenius distance vs gamma2
│   └── gcn_hidden_channels_ablation.py     # 16 vs 32 hidden channels comparison
│
└── quickstart.ipynb                        # Worked examples: running each method on toy data

```

## Install

```bash
pip install -r requirements.txt
```

## Running the code

All scripts use package-relative imports (e.g. `from data_generation.barabasi_albert_datagen import generate_data`),
so they must be run as modules **from the repository root**, not by direct file path:

```bash
# from msc-thesis-code/ (the repo root)
python3 -m simulations.ba_sweep
python3 -m simulations.topology_covariance_sweep
python3 -m diagnostics.gcn_hidden_channels_ablation
```

Running a script by direct path (e.g. `python3 simulations/ba_sweep.py`) will fail with
`ModuleNotFoundError`, since Python adds the script's own folder to `sys.path` rather than the
repo root in that case.

For a minimal example of calling each method directly (without the sweep/checkpointing machinery),
see `quickstart.ipynb`.
