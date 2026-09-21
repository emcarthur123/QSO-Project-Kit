## QSO–LAE extension (Jonas Spiller, ETH Zurich)

The pipeline identifies quasars (QSOs) acting as strong gravitational lenses for background Lyman-α emitters (LAEs) in the DESI DR1 dataset. It builds on the pipeline developed by McArthur et al. (2026), adapting it to higher redshifts and using simulated LAE emission profiles in place of ELG spectra.

It contributes:

- **QSO–LAE mock lenses** — simulated lens spectra created by inserting LAE emission profiles into real DESI quasar spectra, used as the labelled training set
- **An improved CNN classifier** for binary lens / non-lens classification of quasar spectra
- **Genetic-algorithm hyperparameter tuning** of the classifier

> **Note:** This repository does **not** include the LAE augmentation pipeline that generates the LAE spectra used to create the mock lenses. For that, see [augmented_LAE_4QSOlensing](https://github.com/JonasSpiller/augmented_LAE_4QSOlensing) (private, available on request).

### Structure

```bash
advancedML_Pipeline_byJonasSpiller/
├── LensCreation_LAE.ipynb   # Creation of QSO–LAE mock lenses
├── LensCreation_LAE.py      # Creation of QSO–LAE mock lenses (script version)
├── main.ipynb               # Entry point: data loading, model selection and starting the pipeline
├── main.py                  # Main script (script version)
├── utility_functions.py     # Organisation of hyperparameter optimisation and training
├── MLmodels.py              # CNN model architectures
├── HPtuning_GA.py           # Genetic algorithm for hyperparameter optimisation
└── results.ipynb            # Results and model evaluation
```

_Part of this pipeline was inspired by [Emily Garvin's MLCCS](https://github.com/eogarvin/MLCCS)._
