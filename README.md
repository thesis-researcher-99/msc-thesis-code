## Repository Structure

```text
graph-c2st/
├── README.md
│
├── data_generation/                    # Data-generating processes and simulations
│   ├── generate_barabasi_albert.py     # Extended BA model (main simulation + Dubey gamma-based)
│   ├── generate_multi_topology.py      # Ginestet's 5-stage topology → covariance → Laplacian
│   └── generate_lovato.py              # Lovato's data-generating process(es)
│
├── methods/                            # Classifier and statistical testing methods
│   ├── ginestet/
│   ├── dubey/
│   ├── lovato/
│   ├── knn/
│   ├── kernel_svm/
│   └── gcn/
│
├── testing/                            # Hypothesis testing engines
│   ├── permutation_test.py             # Shared engine (KNN/SVM/GCN fixed-prediction permutation)
│   └── fully_designed_permutation_test.py # Lovato's exact TIP-Student/Fisher + Tippett + Phipson-Smyth
│
├── simulations/                        # Pipeline execution and power curve scripts
│   ├── multi_topology_simulation.py    # Ginestet's full grid sweep
│   ├── dubey_simulation.py             # Dubey pipeline simulation
│   └── lovato_simulation.py            # Lovato's power curves execution
│
└── diagnostics/                        # Diagnostic and ablation studies
    ├── gamma_distance_asymmetry.py     # Mean pairwise Frobenius distance vs gamma2 (Fig 18)
    └── gcn_hidden_channels_ablation.py # 16 vs 32 hidden channels comparison
