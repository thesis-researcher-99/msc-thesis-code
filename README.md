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
└── diagnostics/                            # Diagnostic and ablation studies
    ├── ba_distance_asymmetry_check.py         # Mean pairwise Frobenius distance vs gamma2 (Fig X)
    └── gcn_hidden_channels_ablation.py     # 16 vs 32 hidden channels comparison
