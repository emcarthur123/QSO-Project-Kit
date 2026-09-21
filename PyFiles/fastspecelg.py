#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries
import pandas as pd
import fitsio

# Import some helpful python packages 
import os
import numpy as np
from astropy.io import fits
import matplotlib 
import matplotlib.pyplot as plt            # Functions related to coadding the spectra
from desitarget.sv1 import sv1_targetmask    # For SV1 mask
from desitarget.sv2 import sv2_targetmask    # For SV2 mask
from desitarget.sv3 import sv3_targetmask    # For SV3 mask

from astropy.table import Table
import healpy
import h5py# Library for healpix operations


# Directory for spectroscopic product
specprod = 'iron'  # Internal name for the Early Data Release (EDR)
specprod_dir = '/global/cfs/cdirs/desi/public/dr1/spectro/redux/iron'

# Read in the zcatalog file from the specified directory
zpix_cat = Table.read(f'{specprod_dir}/zcatalog/zall-pix-{specprod}.fits', hdu="ZCATALOG")


# Apply filters to select specific galaxies from the catalog
zpix_cat_filtered = zpix_cat[(zpix_cat['ZWARN'] == 0) & 
                             (zpix_cat['SPECTYPE'] == 'GALAXY') &
                             (zpix_cat['OBJTYPE'] == 'TGT') &
                             (zpix_cat['ZCAT_PRIMARY'] == True)]

# Get target columns for different SV (Survey Validation) stages
sv1_desi_tgt = zpix_cat['SV1_DESI_TARGET']
sv2_desi_tgt = zpix_cat['SV2_DESI_TARGET']
sv3_desi_tgt = zpix_cat['SV3_DESI_TARGET']

# Get target masks for SV1, SV2, and SV3
sv1_desi_mask = sv1_targetmask.desi_mask
sv2_desi_mask = sv2_targetmask.desi_mask
sv3_desi_mask = sv3_targetmask.desi_mask

# Select emission line galaxies (ELGs) using target masks for SV1, SV2, and SV3
is_elg = (sv1_desi_tgt & sv1_desi_mask['ELG'] != 0) | (sv2_desi_tgt & sv2_desi_mask['ELG'] != 0) | (sv3_desi_tgt & sv3_desi_mask['ELG'] != 0)
elgs = zpix_cat[is_elg]


# Further filter the ELGs with specific conditions
elgs_new = elgs[(elgs['ZWARN'] == 0) &
                (elgs['SPECTYPE'] == 'GALAXY') &
                (elgs['OBJTYPE'] == 'TGT') &
                (elgs['ZCAT_PRIMARY'] == True)]


# Directory for fastspecfit data
datadir = '/global/cfs/cdirs/desi/spectro/fastspecfit/iron/catalogs'

# Read in the fastspec and metadata files
fast = Table(fitsio.read(os.path.join(datadir, 'fastspec-iron.fits'), 'FASTSPEC'))
meta = Table(fitsio.read(os.path.join(datadir, 'fastspec-iron.fits'), 'METADATA'))


# Create a mask to match the target IDs of ELGs with the fastspec data
mask = np.isin(fast['TARGETID'], elgs_new['TARGETID'])

# Use the mask to index fastspec data and retrieve elements in both fastspec and ELGs
objects_in_both = fast[mask]

# Apply redshift filters
condition_array = (objects_in_both['Z'] >= 0.03) & (objects_in_both['Z'] <= 1.8)
r_unfiltered = objects_in_both[condition_array]

# Further filter by OII emission line flux
r = r_unfiltered[(r_unfiltered['OII_3726_FLUX'] >= 2)]


# Function to retrieve the spectrum for a specific ELG
def Spectrum_elg(jj):
    ind = np.where(meta['TARGETID'] == r['TARGETID'][jj])[0][0]  # Find the index of the target in metadata
    ra = meta['RA'][ind]  # Right ascension
    dec = meta['DEC'][ind]  # Declination
    target_id = meta['TARGETID'][ind]  # Target ID
    program = meta['PROGRAM'][ind]  # Program
    survey = meta['SURVEY'][ind]  # Survey
    healpix = str(healpy.ang2pix(64, ra, dec, lonlat=True, nest=True))  # Get healpix number
    hp_group = healpix[:-2]  # First part of healpix number for grouping

    # Path to the fastspec file for this healpix
    hp_path = f'/global/cfs/cdirs/desi/spectro/fastspecfit/iron/v2.0/healpix/{survey}/{program}/{hp_group}/{healpix}/fastspec-{survey}-{program}-{healpix}.fits.gz'

    # Open the FITS file and retrieve the relevant spectrum
    hp_spectra = fits.open(hp_path)
    hp_ind = np.where(hp_spectra[2].data['TARGETID'] == target_id)[0][0]  # Index of the target in the file

    # Get the continuum and emission data
    continuum = hp_spectra[3].data[hp_ind, 0, :]  # Continuum spectrum
    emission = hp_spectra[3].data[hp_ind, 2, :]  # Emission spectrum
    wavelength = np.arange(3600, 9824 + .8, .8)  # Wavelength array

    # Create a table (as a pandas DataFrame) with wavelength and flux (continuum + emission)
    table_elg = pd.DataFrame({'Wavelength': wavelength, 'Flux': (continuum + emission)})
    
    return table_elg, ind


# Function to process a batch of ELGs
def process_unlensed_batch(start_index, end_index):
    results = []
    for f in range(start_index, end_index):
        try:
            # Retrieve spectrum and index for each ELG in the batch
            ELG_Spec, ind = Spectrum_elg(f)
            targetid_name = str(meta['TARGETID'][ind])  # Target ID name
            redshift_elg = meta['Z'][ind]  # Redshift
            classif = 'ELG'  # Classification as ELG
            results.append((classif, targetid_name, ELG_Spec, redshift_elg))  # Append results
        except Exception as e:
            print(f"Error processing index {f}: {str(e)}. Skipping to the next index.")  # Error handling
    return results


# Main block for batch processing
if __name__ == '__main__':
    # Define the batch size (number of indices per file)
    batch_size = 1500

    # Calculate the number of batches based on the size of the filtered data
    num_batches = len(r) // batch_size

    # Create a directory to store individual files
    output_directory = "fastspec_ELG1"
    os.makedirs(output_directory, exist_ok=True)  # Create directory if it doesn't exist

    # Process and save data in individual files
    for batch_index in range(num_batches + 1):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, len(r))  # Handle the last batch properly

        # Process the current batch
        unlensed_results = process_unlensed_batch(start_index, end_index)

        # Create an individual HDF5 file for each batch (this can be changed to FITS if needed)
        file_name = f"output_data_batch_{batch_index}.h5"
        file_path = os.path.join(output_directory, file_name)

        # Save the results in the HDF5 file
        with h5py.File(file_path, "w") as hdf5_file:
            # Unpack the results into separate arrays
            classifier, name, spectra, redshift = zip(*unlensed_results)

            # Store the data in the HDF5 file
            hdf5_file.create_dataset("classifier", data=np.array(classifier, dtype="S"))
            hdf5_file.create_dataset("name", data=np.array(name, dtype="S"))
            hdf5_file.create_dataset("redshift", data=np.array(redshift))
            hdf5_file.create_dataset("spectra", data=np.array(spectra))

    print(f"Data has been saved to individual HDF5 files in the '{output_directory}' directory.")
