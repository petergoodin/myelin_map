from myelin_map_funcs import dcm_convert, plot_mask_dist, plot_ants_warp, ants_reg, mask_transform, ants_rigid, subj2mni, bias_corr, image_calibration, create_mm_func, image_smooth
import numpy as np
import nibabel as nb
from scipy import stats, signal
import matplotlib.pylab as plt
import seaborn as sns
from nipype.interfaces import ants, dcm2nii
import time
import glob
import multiprocessing as mp

import os


#Number of cores to use
n_cores = mp.cpu_count() -1 #One for the OS.

raw_dir = './raw/'

subj_dirs = next(os.walk(raw_dir))[1]


def mm_mp(subj):


    working_dir = './working/'
    try:
        os.mkdir('./working/')
    except:
        print('Directory {} exists. Not creating'.format('./working/'))
    
    
    out_dir = './out/'
    try:
        os.mkdir('./out/')
    except:
        print('Directory {} exists. Not creating'.format('./out/'))

    working_subj_dir = os.path.join(working_dir, subj)
    
    try:
        os.mkdir(working_subj_dir)
    except:
        print('Directory {} already exists. Not creating.'.format(working_subj_dir))
    
    ##Read in masks
    t1_im_mni_fn = os.path.join('./resources','mni_icbm152_t1_tal_nlin_sym_09a.nii')
    t2_im_mni_fn = os.path.join('./resources','mni_icbm152_t2_tal_nlin_sym_09a.nii')
    
    
    eye_mni_mask_fn = os.path.join('./resources','eye_mask.nii.gz')
    temp_bone_mni_mask_fn = os.path.join('./resources','temp_bone_mask.nii.gz')
    brain_mni_mask_fn = os.path.join('./resources','mni_icbm152_t1_tal_nlin_sym_09a_mask.nii')
    
    patterns = ['*_t1_m*', '*_t2_t*']
    
    scan_dict = {'t1': glob.glob(os.path.join(raw_dir, subj, '*',patterns[0], '*.IMA')),
             't2': glob.glob(os.path.join(raw_dir, subj, '*',patterns[1], '*.IMA'))}
    
    
    
    
    #Start pipeline
    t1_subj_nii, t2_subj_nii = dcm_convert(scan_dict, working_subj_dir) 
    
    reg_output = ants_reg(fixed = t1_subj_nii, moving = t1_im_mni_fn, output_dir = working_subj_dir)
    
    mask_list = [eye_mni_mask_fn, temp_bone_mni_mask_fn, brain_mni_mask_fn]
    
    eye_subj_mask, temp_bone_subj_mask, brain_subj_mask = mask_transform(mask_list = mask_list, 
                                                                         ref = t1_subj_nii, 
                                                                         transmat = reg_output['composite_transform'], 
                                                                         output_dir = working_subj_dir) #Change input to reg_results
     
    plot_ants_warp(fixed = t1_subj_nii, moving = brain_subj_mask, nslices = 10, output_name = os.path.join(working_subj_dir, 'brain_mask'))
    
    rigid_output = ants_rigid(fixed = t1_subj_nii, moving = t2_subj_nii, output_dir = working_subj_dir)
    
    t1_bias, t2_bias = bias_corr([t1_subj_nii, rigid_output['warped_image']], output_dir = working_subj_dir)
    
    t1_cal, t2_cal = image_calibration(t1_subj_bias_corr = t1_bias, 
                                                           t2_subj_bias_corr = t2_bias, 
                                                           t1_mni_bias_corr = t1_im_mni_fn, 
                                                           t2_mni_bias_corr = t2_im_mni_fn, 
                                                           eye_subj_mask = eye_subj_mask, 
                                                           temp_bone_subj_mask = temp_bone_subj_mask, 
                                                           brain_subj_mask = brain_subj_mask, 
                                                           eye_mni_mask = eye_mni_mask_fn, 
                                                           temp_bone_mni_mask = temp_bone_mni_mask_fn, 
                                                           brain_mni_mask = brain_mni_mask_fn, 
                                                           output_directory = working_subj_dir)
    
    myelin_map_subj = create_mm_func(t1_cal, t2_cal, output_dir = working_subj_dir)
    
    
    subj2mni_output, subj2mni_im = subj2mni(moving = myelin_map_subj, ref = t1_im_mni_fn, transmat = reg_output['inverse_composite_transform'], output_directory = working_subj_dir) 
    
    for im in [myelin_map_subj, subj2mni_im]:
        smoothed = image_smooth(image_fn = im, fwhm = 2, output_dir = working_subj_dir)


if __name__ == '__main__':
    pool = mp.Pool(processes = n_cores)
    pool.map(mm_mp, subj_dirs)





































































pool = mp.Pool(3)

pool
