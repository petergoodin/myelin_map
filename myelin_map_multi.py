from myelin_map_funcs import dcm_convert, plot_mask_dist, plot_ants_warp, ants_reg, mask_transform, ants_rigid, subj2mni, bias_corr, image_calibration, create_mm_func, image_smooth, mm_minmax
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

raw_dir = './tia_t1/'

subj_dirs = next(os.walk(raw_dir))[1]
# subj_dirs = ['TIA23B_0928214', 'TIA25B_0670296', 'TIA26A_0198263', 'TIA26B_0198263']

print('Number of subjects to be processed: {}'.format(len(subj_dirs)))


def mm_mp(subj):

    print('Processing data for subj {}'.format(subj))

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
    print(working_subj_dir)

    try:
        os.mkdir(working_subj_dir)
    except:
        print('Directory {} already exists. Not creating.'.format(working_subj_dir))

    print("\nWorking dir: {}".format(working_subj_dir))

    ##Read in masks
    t1_im_mni_fn = os.path.join('./resources','mni_icbm152_t1_tal_nlin_sym_09a.nii')
    t2_im_mni_fn = os.path.join('./resources','mni_icbm152_t2_tal_nlin_sym_09a.nii')


    eye_mni_mask_fn = os.path.join('./resources','eye_mask.nii.gz')
    temp_bone_mni_mask_fn = os.path.join('./resources','temp_bone_mask.nii.gz')
    # brain_mni_mask_fn = os.path.join('./resources','mni_icbm152_t1_tal_nlin_sym_09a_mask.nii')
    brain_mni_mask_fn = os.path.join('./resources','matter_mask.nii.gz')

    patterns = ['*_t1_m*', '*_t2_f*', '*_t2_t*']

    scan_dict = {'t1': glob.glob(os.path.join(raw_dir, subj, '*',patterns[0], '*.IMA')),
                 't2': []}

    if 'A_' in subj:
        scan_dict['t2'] = glob.glob(os.path.join(raw_dir, subj, '*', patterns[1], '*.IMA'))
        if len(scan_dict['t2']) < 40:
            t2_list = glob.glob(os.path.join(raw_dir, subj, '*', patterns[2]))
            t2_dir = t2_list[0]
            scan_dict['t2'] = glob.glob(os.path.join(t2_dir, '*.IMA'))
    else:
        t2_list = glob.glob(os.path.join(raw_dir, subj, '*', patterns[2]))
        t2_dir = t2_list[0]
        #The first T2 scan is the non fluid suppressed
        scan_dict['t2'] = glob.glob(os.path.join(t2_dir, '*.IMA'))

    im_count = {k: len(list(scan_dict.values())[n]) for n, k in enumerate(scan_dict)}
    print(subj, im_count)

    with open(os.path.join(working_subj_dir, subj + 'input_files_n.txt'), 'w') as text_file:
        text_file.write('{} - {}'.format(subj, im_count))


    #Generate error if data not matching criteria
    if len(scan_dict['t1']) < 10:
        with open(os.path.join(working_subj_dir, subj + '_t1_error.txt'), 'w') as text_file:
            text_file.write('Number of T1 scans < 100: {}'.format(len(scan_dict['t1'])))
        #raise ValueError('Error: Number of DICOMS in T1 directory less than expected for subj {}\n# of Dicoms = {}'.format(subj, len(scan_dict['t1'])))

    if len(scan_dict['t2']) < 10:
        with open(os.path.join(working_subj_dir, subj + '_t2_error.txt'), 'w') as text_file:
            text_file.write('Number of T2 scans < 40: {}'.format(len(scan_dict['t2'])))
        #raise ValueError('Error: Number of DICOMS in T2 directory less than expected for subj {}\n# of Dicoms = {}'.format(subj, len(scan_dict['t2'])))


    #Start pipeline
    t1_subj_nii, t2_subj_nii = dcm_convert(scan_dict, working_subj_dir)

    reg_output = ants_reg(fixed = t1_subj_nii, moving = t1_im_mni_fn, output_dir = working_subj_dir)

    mask_list = [eye_mni_mask_fn, temp_bone_mni_mask_fn, brain_mni_mask_fn]

    eye_subj_mask, temp_bone_subj_mask, brain_subj_mask = mask_transform(mask_list = mask_list,
                                                                         ref = t1_subj_nii,
                                                                         transmat = reg_output['composite_transform'],
                                                                         output_dir = working_subj_dir)

#    plot_ants_warp(fixed = t1_subj_nii, moving = brain_subj_mask, nslices = 10, output_name = os.path.join(working_subj_dir, 'brain_mask'))

    rigid_output = ants_rigid(fixed = t1_subj_nii, moving = t2_subj_nii, output_dir = working_subj_dir)

    t1_bias, t2_bias = bias_corr([t1_subj_nii, rigid_output['warped_image']], brain_mask = brain_subj_mask, output_dir = working_subj_dir)

    #Calibration - TO DO: Split function into mode calculation + linear correction
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
                                                           output_dir = working_subj_dir)

    #Calculate myelin maps
    myelin_map_subj = create_mm_func(t1_cal, t2_cal, brain_mask = brain_subj_mask, output_dir = working_subj_dir)

    #Warp myelin maps to MNI space
    subj2mni_output, subj2mni_im = subj2mni(moving = myelin_map_subj, ref = t1_im_mni_fn, transmat = reg_output['inverse_composite_transform'], output_dir = working_subj_dir)

    #Smooth
    for im in [myelin_map_subj, subj2mni_im]:
        for fwhm in [1,2]:
            smoothed = image_smooth(image_fn = im, fwhm = fwhm, output_dir = working_subj_dir)

    # #Thresh
    # mmap_list = glob.glob(working_subj_dir + '/myelin_map*')
    # mmap_list = [im for im in mmap_list if 'minmax' not in im]
    # print(mmap_list)
    #
    # for im in mmap_list:
    #     if 'mni' in im:
    #         mm_minmax(mmap = im, mask = brain_mni_mask_fn, output_dir = working_subj_dir)
    #     else:
    #         mm_minmax(mmap = im, mask = brain_subj_mask, output_dir = working_subj_dir)

if __name__ == '__main__':
    pool = mp.Pool(processes = n_cores)
    pool.map(mm_mp, subj_dirs)
