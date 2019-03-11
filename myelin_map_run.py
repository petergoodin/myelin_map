from myelin_map_funcs import dcm_convert, plot_mask_dist, plot_ants_warp, ants_reg, mask_transform, ants_rigid, subj2mni, bias_corr, image_calibration, create_mm_func, image_smooth, mm_percentile, myelin_map_proc
import numpy as np
import nibabel as nb
from scipy import stats, signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from nipype.interfaces import ants, dcm2nii
import time
import glob
import multiprocessing as mp
import joblib
import os


###Parameters###

#Number of cores to use
n_cores = mp.cpu_count() -1 #One for the OS.

#Path to directory which contains subj/t1 and subj/t2 directories
raw_dir = '/mnt/c/Users/pgoodin/Desktop/mm_test/raw'


#Path to root output directory (note: subj directories will be created as part of processing).
output_dir = '/mnt/c/Users/pgoodin/Desktop/mm_test/output'

#Collect list of participant names (three methods, use which ever works best)
subj_dirs = [subj for subj in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, subj))]
#subj_dirs = next(os.walk(raw_dir))[1]
#subj_dirs = ['test_subj'] #Insert list of participant directory names here
print('Number of subjects to be processed: {}'.format(len(subj_dirs)))

#Strings to check for to determine t1 / t2 directories. Note: Uses glob, so wildcards / single string placeholders are acceptable.
patterns = ['t1_dir', 't2_dir']

#Dicom suffix
dcm_suffix = '.dcm'

#List of smoothing kernel sizes (in mm) - Note: Can be single or multiple values.
fwhm_list = [1, 2, 3]

#Number of expected scans for t1 and t2:
n_scans = {'t1': 100, 't2': 100}



###Here the processing begins###

if len(subj_dirs) == 1:
    myelin_map_proc(subj = subj_dirs[0], n_cores = n_cores, raw_dir = raw_dir, output_dir = output_dir, patterns = patterns, n_scans = n_scans, dcm_suffix = dcm_suffix, fwhm_list = fwhm_list)

elif len(subj_dirs) > 1:
    joblib.Parallel(n_jobs = n_cores, verbose = 10)(joblib.delayed(myelin_map_proc)(subj = subj, n_cores = n_cores, raw_dir = raw_dir, output_dir = output_dir, patterns = patterns, n_scans = n_scans, dcm_suffix = dcm_suffix, fwhm_list = fwhm_list) for subj in subj_dirs)

else:
    raise ValueError('Error: No participant names found in subj_dirs. Please check')
