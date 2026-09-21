#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import os
from astropy.io import fits
from astropy.table import Table
import desisim.templates
import desispec.io

specprod = 'iron'    # Internal name for the EDR
specprod_dir = '/global/cfs/cdirs/desi/public/dr1/spectro/redux/iron'

zpix_cat = Table.read(f'{specprod_dir}/zcatalog/zall-pix-{specprod}.fits', hdu="ZCATALOG"
                     )

Quasar_cat2 = zpix_cat[(zpix_cat['SPECTYPE'] == 'QSO') & 
     (zpix_cat['Z'] < 1.8) & 
     (zpix_cat['Z'] >= 0.03) & 
     (zpix_cat['ZWARN'] == 0 ) & 
     (zpix_cat['OBJTYPE'] == 'TGT') & 
     (zpix_cat['ZCAT_PRIMARY'] == 1) &
     (zpix_cat['SURVEY']=='main') & (zpix_cat['PROGRAM']=='dark')]

num = np.unique(Quasar_cat2['HEALPIX'])

from desispec.spectra import stack

# Initialize variables
num = np.unique(Quasar_cat2['HEALPIX'])
test_array = []
targetids_array = []
z_array = []
stack_count = 0
batch_size = 50  # Number of HEALPIX values to stack and save at once

# Loop through each unique HEALPIX value
for i in range(len(num)):
    try:
        # Filter for the specific HEALPIX value in this iteration
        Quasar_cat_filtered = Quasar_cat2[Quasar_cat2['HEALPIX'] == num[i]]
        
        if len(Quasar_cat_filtered) == 0:
            print(f"No quasar data for HEALPIX {num[i]}")
            continue

        survey = Quasar_cat_filtered["SURVEY"][0]
        program = Quasar_cat_filtered["PROGRAM"][0]
        hpx = Quasar_cat_filtered["HEALPIX"][0]

        specprod_dir = f"/global/cfs/cdirs/desi/spectro/redux/{specprod}"
        target_dir = f"{specprod_dir}/healpix/{survey}/{program}/{hpx//100}/{hpx}"
        coadd_fname = f"coadd-{survey}-{program}-{hpx}.fits"

        # Read spectra data
        spectra = desispec.io.read_spectra(f"{target_dir}/{coadd_fname}", skip_hdus=('EXP_FIBERMAP', 'SCORES', 'EXTRA_CATALOG')).select(targets=Quasar_cat_filtered['TARGETID'])
        
        # Append to test_array for stacking and targetids and z for saving
        test_array.append(spectra)
        targetids_array.extend(Quasar_cat_filtered['TARGETID'])
        z_array.extend(Quasar_cat_filtered['Z'])
        stack_count += 1
        
        # Check if it's time to stack and save
        if stack_count == batch_size or i == len(num) - 1:
            # Stack spectra
            stacked = stack(test_array)
            
            # Sort both the spectra and the additional data by TARGETID to ensure they are row-matched
            ii = np.argsort(stacked.fibermap['TARGETID'])
            stacked = stacked[ii]

            jj = np.argsort(targetids_array)
            targetids_array_sorted = np.array(targetids_array)[jj]
            z_array_sorted = np.array(z_array)[jj]

            # Ensure the TARGETIDs match
            assert np.all(stacked.fibermap['TARGETID'] == targetids_array_sorted)

            # Create an astropy Table for the extra catalog
            extra_catalog = Table()
            extra_catalog['TARGETID'] = targetids_array_sorted
            extra_catalog['Z'] = z_array_sorted
            
            # Attach the extra catalog to the stacked spectra
            stacked.extra_catalog = extra_catalog

            # Output file for this batch
            batch_number = i // batch_size + 1
            outfile = os.path.expandvars(f'$SCRATCH/MainQSO/desi_bright_qso_batch{batch_number}.fits')
            
            # Write stacked spectra to file
            desispec.io.write_spectra(outfile, stacked)
            
            print(f"Batch {batch_number} saved to {outfile}")
            
            # Reset variables for the next batch
            test_array = []
            targetids_array = []
            z_array = []
            stack_count = 0

    except (FileNotFoundError, IOError) as e:
        print(f"Error reading file for index {i} (HEALPIX {num[i]}): {e}")
        continue
