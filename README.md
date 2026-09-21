# QSO-Project-Kit

> **Fork notice.** This repository is a fork of Everett McArthur's work on simulating and classifying QSO–ELG strong gravitational lenses in DESI, published as McArthur et al. (2026), *The Astrophysical Journal*. My contribution, in [`advancedML_Pipeline_byJonasSpiller/`](advancedML_Pipeline_byJonasSpiller/), extends the search to QSO–LAE lenses by simulating QSO–LAE mock lenses and designing an improved CNN classifier with a genetic algorithm for hyperparameter tuning. This extension has since been merged into McArthur's upstream repository, which underlies McArthur et al. (2026); the QSO–LAE search itself is a follow-on extension and is not part of that paper. Everything outside that directory is McArthur's original work and is documented [below](#original-project-qsolensefinder-everett-mcarthur).

---

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

---

## Original project: QSOLenseFinder (Everett McArthur)

_The following is McArthur's original README, reproduced unchanged._

QSOLenseFinder is a tool designed to determine whether or not QSOs (Quasi-Stellar Objects) in the DESI (Dark Energy Spectroscopic Instrument) survey are strong gravitational lenses. This repository contains a Python implementation of the model using Keras version 2.13.1 and NumPy version 1.24.3, as well as example notebooks to get you started on how to use the desi redshift finder, fastspec, and accessing desi spectra. Below are also relevant papers to get you started.

FastSpec
https://ui.adsabs.harvard.edu/abs/2023ascl.soft08005M/abstract

Redrock
https://iopscience.iop.org/article/10.3847/1538-3881/acb212

DESI Getting Started Github
https://github.com/desihub/tutorials/tree/main

### Overview

This project includes a neural network architecture and prediction function to classify QSOs based on their spectra. The model architecture uses a series of convolutional layers followed by fully connected layers to perform binary classification.

### Dependencies

- Keras 2.13.1
- NumPy 1.24.3
- Matplotlib
- Scipy
- Pandas
- Scikit
- Astropy

### Installation

To use this project, ensure you have the required versions of Keras and NumPy installed. You can install them using pip
Also be sure you are a member of DESI to gain access to the nersc file system.

---

## Citation

If you use this code, please cite:

McArthur, E., et al. (2026). "Quasars Acting as Strong Lenses Found in DESI DR1." *The Astrophysical Journal*.
