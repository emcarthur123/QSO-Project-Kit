# QSOLenseFinder

QSOLenseFinder is a tool designed to determine whether or not QSOs (Quasi-Stellar Objects) in the DESI (Dark Energy Spectroscopic Instrument) survey are strong gravitational lenses. This repository contains a Python implementation of the model using Keras version 2.13.1 and NumPy version 1.24.3, as well as example notebooks to get you started on how to use the desi redshift finder, fastspec, and accessing desi spectra. Below are also relevant papers to get you started.

FastSpec
https://ui.adsabs.harvard.edu/abs/2023ascl.soft08005M/abstract

Redrock
https://iopscience.iop.org/article/10.3847/1538-3881/acb212

DESI Getting Started Github
https://github.com/desihub/tutorials/tree/main

## Overview

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
