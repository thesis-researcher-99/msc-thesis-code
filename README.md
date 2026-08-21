# msc-thesis-code
code used in thesis

msc-thesis-code/
├── README.md
│
├── data_generation/
│   ├── generate_barabasi_albert.py        # Extended BA (main simulation + Dubey gamma-based)
│   ├── generate_multi_topology.py         # Ginestet's 5-stage topology→covariance→Laplacian
│   └── generate_lovato.py                 # Lovato's data-generating process(es)
│
├── methods/
│   ├── ginestet/
│   ├── dubey/
│   ├── lovato/
│   ├── knn/
│   ├── kernel_svm/
│   └── gcn/
│
├── testing/
│   ├── permutation_test.py                # shared engine (KNN/SVM/GCN fixed-prediction permutation)
│   └── fully_designed_permutation_test.py # Lovato's exact TIP-Student/Fisher + Tippett + Phipson-Smyth
│
├── simulations/
│   ├── multi_topology_simulation.py       # Ginestet's full grid sweep
│   ├── dubey_simulation.py (rename pending) 
│   └── lovato_simulation.py               # whatever runs Lovato's power curves
│
└── diagnostics/
    ├── gamma_distance_asymmetry.py        # mean pairwise Frobenius distance vs gamma2 (Fig 18)
    └── gcn_hidden_channels_ablation.py    # 16 vs 32 hidden channels comparison
