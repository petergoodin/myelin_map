from __future__ import division
import numpy as np
import nibabel as nb
from nibabel import processing as nbproc
from scipy import stats, signal
import matplotlib.pylab as plt
import seaborn as sns
from nipype.interfaces import ants, dcm2nii
import time
import glob
import multiprocessing as mp

import os


def dcm_convert(scan_dict, output_dir):

    scan_list = list(scan_dict.keys())
    anat_convert_output = []

    for scan in scan_list:

        anat_convert = dcm2nii.Dcm2niix()
        anat_convert.inputs.source_names = scan_dict[scan]
        anat_convert.inputs.out_filename = scan
        anat_convert.inputs.output_dir = output_dir
        anat_convert.inputs.compress = 'y'
        results = anat_convert.run()

        anat_convert_output.append(results.outputs.get()['converted_files'])
    t1_subj_nii, t2_subj_nii = anat_convert_output

    return(t1_subj_nii, t2_subj_nii)


###PLOTTING###

#Plot histograms of data
def plot_mask_dist(t1_fn, t2_fn, eye_mask_fn, temp_bone_mask_fn, stat = None):

    t1 = nb.load(t1_fn).get_data().ravel()
    t2 = nb.load(t2_fn).get_data().ravel()

    eye_mask = nb.load(eye_mask_fn).get_data().ravel()
    temp_mask = nb.load(temp_bone_mask_fn).get_data().ravel()


    if stat == 'mean':
        t1_eye_stat = np.mean(t1[eye_mask])
        t1_temp_stat = np.mean(t1[temp_mask])
        t2_eye_stat = np.mean(t2[eye_mask])
        t2_temp_stat = np.mean(t2[temp_mask])

    elif stat == 'median':
        t1_eye_stat = np.median(t1[eye_mask])
        t1_temp_stat = np.median(t1[temp_mask])
        t2_eye_stat = np.median(t2[eye_mask])
        t2_temp_stat = np.median(t2[temp_mask])


    elif stat == 'mode' or stat == None:
        t1_eye_stat = stats.mode(t1[eye_mask][t1[eye_mask] > 0])[0][0]
        t1_temp_stat = stats.mode(t1[temp_mask][t1[temp_mask] > 0])[0][0]
        t2_eye_stat = stats.mode(t2[eye_mask][t2[eye_mask] > 0])[0][0]
        t2_temp_stat = stats.mode(t2[temp_mask][t2[temp_mask] > 0])[0][0]


    fig = plt.figure(figsize = [15, 10]);
    plt.subplot(2,1,1);
    plt.title('Eye ' + stat)
    sns.distplot(t1[eye_mask], label = 'T1');
    ymin, ymax = fig.gca().axes.get_ybound()
    plt.vlines(x = t1_eye_stat, ymin = ymin, ymax = ymax)
    sns.distplot(t2[eye_mask], label = 'T2');
    plt.vlines(x = t2_eye_stat, ymin = ymin, ymax = ymax)
    plt.legend()
    plt.show()

    fig = plt.figure(figsize = [15, 10]);
    plt.subplot(2,1,2);
    plt.title('Temporal Bone ' + stat)
    sns.distplot(t1[temp_mask], label = 'T1');
    ymin, ymax = fig.gca().axes.get_ybound()
    plt.vlines(x = t1_temp_stat, ymin = ymin, ymax = ymax)
    sns.distplot(t2[temp_mask], label = 'T2');
    plt.vlines(x = t2_temp_stat, ymin = ymin, ymax = ymax)
    plt.legend()
    plt.show()

    print('T1 eye: {}\nT2 eye: {}\nT1 temp: {}\nT2 temp: {}'.format(t1_eye_stat, t2_eye_stat, t1_temp_stat, t2_temp_stat))


#Plot image overlays to assess warp quality
def plot_ants_warp(fixed, moving, nslices, output_name = None):
    """
    Plots nslices axial images of the fixed and moving images from the ants transform

    inputs:
    anat - Fixed image (image the moving was warped to)
    moving - Moving image (Transformed image)
    nslices - Number of slices to plot
    output_name - filename to save png image to (optional)

    """
    fixed_data = nb.load(fixed).get_data()
    moving_data = nb.load(moving).get_data()

    view_slices = np.linspace(100, fixed_data.shape[1] - 1, num = nslices).astype(int)

    fig = plt.figure(figsize = [50, 25])

    for n, view_slice in enumerate(view_slices):
        plt.subplot(1, nslices, n + 1)
        plt.imshow(fixed_data[:, :, view_slice], cmap = 'Greys_r')
        plt.imshow(moving_data[:, :, view_slice], cmap = 'Reds', alpha = 0.25)
        plt.axis('off')
        plt.text(1,1, 'z = ' + str(view_slice), color = [1,0,0], bbox=dict(facecolor=[0,0,0]), fontsize = 20)
        plt.tight_layout()
    plt.show()

    if output_name != None:
        fig.savefig(fname = output_name + '.png')



###SPACIAL TRANSORMS###

#Warp MNI to subj
def ants_reg(fixed = None, moving = None, prefix = None, fixed_mask = None, moving_mask = None, output_dir = None):

    '''
    Uses ANTs to warp the moving image to the space of the fixed.
    Both fixed and moving images can have masks.

    If warping subjects with lesion damage, it is recommended to warp from MNI to subj space
    (fixed = subj, moving = mni template, fixed_mask = lesion mask) then
    use the inverse transform to warp from subj to MNI.

    '''
    reg = ants.Registration()
    reg.inputs.fixed_image = fixed
    reg.inputs.moving_image = moving

    if fixed_mask != None:
        reg.inputs.fixed_mask = fixed_mask

    if moving_mask != None:
        reg.inputs.moving_mask = fixed_mask

    if prefix != None:
        reg.inputs.output_transform_prefix = prefix
    else:
        reg.inputs.output_transform_prefix = os.path.join(output_dir, 'reg_trans_')

    reg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    reg.inputs.transform_parameters = [(0.1,),(0.1,),(0.1, 3.0, 0.0)] #Size of movement for registration (Optimal values are 0.1-0.25.)
    reg.inputs.number_of_iterations =[[1000, 500, 250, 100],[1000, 500, 250, 100],[100, 70, 50, 20]]
    reg.inputs.dimension = 3
    # '''
    # Align the moving_image and fixed_image before registration using the geometric
    # center of the images (=0), the image intensities (=1),or the origin of the images (=2)
    # '''
    reg.inputs.initial_moving_transform_com = 0
    reg.inputs.write_composite_transform = True
    reg.inputs.collapse_output_transforms = False
    reg.inputs.initialize_transforms_per_stage = False
    reg.inputs.metric = ['MI', 'MI', 'CC']
    reg.inputs.radius_or_number_of_bins = [32, 32, 4]
    reg.inputs.sampling_strategy = ['Regular','Regular','None']
    reg.inputs.sampling_percentage = [0.25, 0.25, 1]
    reg.inputs.convergence_threshold = [1e-06]
    reg.inputs.convergence_window_size = [10]
    reg.inputs.smoothing_sigmas = [[3, 2, 1, 0]] * 3
    reg.inputs.sigma_units = ['vox'] * 3
    reg.inputs.shrink_factors = [[8, 4, 2, 1]] * 3
    reg.inputs.use_estimate_learning_rate_once = [True, True, True]
    reg.inputs.use_histogram_matching = True

    if output_dir != None:
        reg.inputs.output_warped_image = os.path.join(output_dir, 'output_warped_image.nii.gz')
    else:
        reg.inputs.output_warped_image = './output_warped_image.nii.gz'
   
    reg.inputs.num_threads = 1
    reg.inputs.metric_weight = [1.0] * 3
    reg.inputs.winsorize_lower_quantile = 0.005
    reg.inputs.winsorize_upper_quantile = 0.995
    reg.inputs.verbose = True

    reg_results = reg.run()
    
    reg_out_files = reg_results.outputs.get()
    
    return(reg_out_files)


#Transform eye + temporal mask + brain mask from MNI to subj
def mask_transform(mask_list, ref, transmat, output_dir):
    '''
    Transforms masks from MNI to subj space
    masks = list of eye + temporal mask images
    transmat = mapping from MNI > subj space
    '''

    subj_trans_masks = []

    for mask in mask_list:

        image_file = os.path.split(mask)[1]
        image_name = image_file.split('.')[0]
        print(image_name)

        mni2subj = ants.ApplyTransforms()
        mni2subj.inputs.input_image = mask
        mni2subj.inputs.reference_image = ref
        mni2subj.inputs.transforms = transmat
        mni2subj.inputs.interpolation = 'NearestNeighbor'
        mni2subj.inputs.output_image = os.path.join(output_dir, image_name + '_subj.nii.gz')
        mni2subj_results = mni2subj.run()
        
        output_image = mni2subj_results.outputs.get()['output_image']

        subj_trans_masks.append(output_image)

    return(subj_trans_masks)




#Register T2 to T1
def ants_rigid(fixed = None, moving = None, prefix = None, fixed_mask = None, moving_mask = None, output_dir = None):
    rigid = ants.Registration()

    rigid.inputs.fixed_image = fixed
    rigid.inputs.moving_image = moving
    
    if fixed_mask != None:
        rigid.inputs.fixed_mask = fixed_mask

    if moving_mask != None:
        rigid.inputs.moving_mask = fixed_mask

    if prefix != None:
        rigid.inputs.output_transform_prefix = prefix
    else:
        rigid.inputs.output_transform_prefix = os.path.join(output_dir, 'rigid_trans_')
    
    
    rigid.inputs.transforms = ['Rigid']
    rigid.inputs.transform_parameters = [(0.1,)] #Size of movement for registration (Optimal values are 0.1-0.25.)
    rigid.inputs.number_of_iterations = [[1000, 500, 250, 100]]
    rigid.inputs.dimension = 3
    #
    # Align the moving_image and fixed_image before registration using the geometric
    # center of the images (=0), the image intensities (=1),or the origin of the images (=2)
    #
    rigid.inputs.initial_moving_transform_com = 0
    rigid.inputs.write_composite_transform = True
    rigid.inputs.collapse_output_transforms = False
    rigid.inputs.initialize_transforms_per_stage = False
    rigid.inputs.metric = ['MI']
    rigid.inputs.radius_or_number_of_bins = [32]
    rigid.inputs.sampling_strategy = ['Regular']
    rigid.inputs.sampling_percentage = [0.25]
    rigid.inputs.convergence_threshold = [1e-06]
    rigid.inputs.convergence_window_size = [10]
    rigid.inputs.smoothing_sigmas = [[3, 2, 1, 0]]
    rigid.inputs.sigma_units = ['vox']
    rigid.inputs.shrink_factors = [[8, 4, 2, 1]]
    rigid.inputs.use_estimate_learning_rate_once = [True]
    rigid.inputs.use_histogram_matching = [True]
    
    
    if output_dir != None:
        rigid.inputs.output_warped_image = os.path.join(output_dir, 'output_rigid_image.nii.gz')
    else:
        rigid.inputs.output_warped_image = './output_rigid_image.nii.gz'
    
    
    rigid.inputs.num_threads = 8
    rigid.inputs.metric_weight = [1.0]
    rigid.inputs.winsorize_lower_quantile = 0.005
    rigid.inputs.winsorize_upper_quantile = 0.995
    rigid.inputs.verbose = True
    rigid.inputs.interpolation = 'NearestNeighbor' #Started with NN, gave good output.
    rigid_results = rigid.run()
    
    rigid_out_files = rigid_results.outputs.get()
    
    return(rigid_out_files)

#Myelin map to MNI
def subj2mni(moving = None, ref = None, transmat = None, output_directory = None):

    subj2mni = ants.ApplyTransforms()
    subj2mni.inputs.dimension = 3
    subj2mni.inputs.input_image = moving
    subj2mni.inputs.reference_image = ref
    subj2mni.inputs.transforms = transmat


    if output_directory != None:
        subj2mni.inputs.output_image = os.path.join(output_directory, 'myelin_map_mni.nii.gz')
    else:
        subj2mni.inputs.output_image = ('myelin_map_mni.nii.gz')
    subj2mni.inputs.interpolation = 'NearestNeighbor'
    subj2mni_results = subj2mni.run()
    
    subj2mni_output = subj2mni_results.outputs.get()
    subj2mni_im = subj2mni_output['output_image']
    
    return(subj2mni_output, subj2mni_im)


###INTENSITY MANIPULATION###

#Bias correction
def bias_corr(images, output_dir):
    '''
    Uses N4 bias correction to remove intensity inhomogeneities

    Input:
    images = List of images (w/ paths) to be corrected
    '''

    bias_output = []

    for n, image in enumerate(images):
#        t1 = time.time()
        image_file = os.path.split(image)[1]
        image_name = image_file.split('.')[0]
        print(image, image_name)

        n4  = ants.N4BiasFieldCorrection()
        n4.inputs.dimension = 3
        n4.inputs.input_image = image
        n4.inputs.num_threads = 1
        n4.inputs.bspline_fitting_distance = 300
        n4.inputs.shrink_factor = 2
        n4.inputs.n_iterations = [50,50,30,20]
        n4.inputs.save_bias = True
        n4.inputs.bias_image = os.path.join(output_dir, image_name + '_bias_field.nii.gz')
        n4.inputs.output_image = os.path.join(output_dir, image_name + '_bias_corr.nii.gz')
        n4_results = n4.run()
        
        output_image = n4_results.outputs.get()['output_image']

        bias_output.append(output_image)
    
    return(bias_output)


def image_smooth(image_fn, fwhm = 1, output_dir = None):
    
    im_name = os.path.split(image_fn)[-1].split('.')[0]
    
    im_hdr = nb.load(image_fn)
    
    smoothed = nbproc.smooth_image(img = im_hdr, fwhm = fwhm)
    
    output_fn = os.path.join(output_dir, im_name + '_smoothed_' + str(fwhm) + '_mm.nii.gz')
    
    nb.Nifti1Image(smoothed.get_data(), affine = im_hdr.affine, header = im_hdr.header).to_filename(output_fn)
    
    return(output_fn)



#Calibration stage
def image_calibration(t1_subj_bias_corr = None, t2_subj_bias_corr = None, t1_mni_bias_corr = None, t2_mni_bias_corr = None, eye_subj_mask = None, temp_bone_subj_mask = None, brain_subj_mask = None, eye_mni_mask = None, temp_bone_mni_mask = None, brain_mni_mask = None, output_directory = None):

    #load images
    t1_subj_hdr = nb.load(t1_subj_bias_corr)
    t1_subj = t1_subj_hdr.get_data()
    t2_subj = nb.load(t2_subj_bias_corr).get_data()
    
    t1_mni = nb.load(t1_mni_bias_corr).get_data()
    t2_mni = nb.load(t2_mni_bias_corr).get_data()

    #Load masks
    eye_subj_mask = nb.load(eye_subj_mask).get_data().astype(bool)
    temp_bone_subj_mask = nb.load(temp_bone_subj_mask).get_data().astype(bool)
    brain_subj_mask = nb.load(brain_subj_mask).get_data().astype(bool)
    
    eye_mni_mask = nb.load(eye_mni_mask).get_data().astype(bool)
    temp_bone_mni_mask = nb.load(temp_bone_mni_mask).get_data().astype(bool)
    brain_mni_mask = nb.load(brain_mni_mask).get_data().astype(bool)
    

    #Extract stats
    t1_subj_eye = stats.mode(t1_subj[eye_subj_mask][t1_subj[eye_subj_mask] > 0], axis = None)[0][0]
    t1_subj_temp = stats.mode(t1_subj[temp_bone_subj_mask][t1_subj[temp_bone_subj_mask] > 0], axis = None)[0][0]
    t2_subj_eye = stats.mode(t2_subj[eye_subj_mask][t2_subj[eye_subj_mask] > 0], axis = None)[0][0]
    t2_subj_temp = stats.mode(t2_subj[temp_bone_subj_mask][t2_subj[temp_bone_subj_mask] > 0], axis = None)[0][0]
    
    t1_mni_eye = stats.mode(t1_mni[eye_mni_mask][t1_mni[eye_mni_mask] > 0], axis = None)[0][0]
    t1_mni_temp = stats.mode(t1_mni[temp_bone_mni_mask][t1_mni[temp_bone_mni_mask] > 0], axis = None)[0][0]
    t2_mni_eye = stats.mode(t2_mni[eye_mni_mask][t2_mni[eye_mni_mask] > 0], axis = None)[0][0]
    t2_mni_temp = stats.mode(t2_mni[temp_bone_mni_mask][t2_mni[temp_bone_mni_mask] > 0], axis = None)[0][0]

    #Shorten linear equation for easier troubleshooting
    t1_a = (t1_mni_temp - t1_mni_eye) / (t1_subj_temp - t1_subj_eye)
    t1_b = ((t1_subj_temp * t1_mni_eye) - (t1_mni_temp * t1_subj_eye)) / (t1_subj_temp - t1_subj_eye)

    t2_a = (t2_mni_temp - t2_mni_eye) / (t2_subj_temp - t2_subj_eye)
    t2_b = ((t2_subj_temp * t2_mni_eye) - (t2_mni_temp * t2_subj_eye)) / (t2_subj_temp - t2_subj_eye)


    #Intensity correction
    t1_corr = (t1_a * t1_subj[brain_subj_mask]) + t1_b
    t2_corr = (t2_a * t2_subj[brain_subj_mask]) + t2_b

    print('{} {}'.format(t1_subj.min(), t1_subj.max()))
    print('{} {}'.format(t1_subj.min(), t1_subj.max()))

    print('{} {}'.format(t1_corr.min(), t1_corr.max()))
    print('{} {}'.format(t2_corr.min(), t2_corr.max()))
    
    t1_corr_out = np.zeros_like(t1_subj)
    t2_corr_out = np.zeros_like(t2_subj)

    t1_corr_out[brain_subj_mask] = t1_corr
    t2_corr_out[brain_subj_mask] = t2_corr

    if output_directory != None:
        t1_fn = os.path.join(output_directory, 't1_calibrated.nii.gz')
        t2_fn = os.path.join(output_directory, 't2_calibrated.nii.gz')
    else:
        t1_fn = 't1_calibrated.nii.gz'
        t2_fn = 't2_calibrated.nii.gz'


    nb.Nifti1Image(t1_corr_out, affine = t1_subj_hdr.affine, header = t1_subj_hdr.header).to_filename(t1_fn)
    nb.Nifti1Image(t2_corr_out, affine = t1_subj_hdr.affine, header = t1_subj_hdr.header).to_filename(t2_fn)

    return(t1_fn, t2_fn)



###Myelin Map###

def create_mm_func(corrected_t1, corrected_t2, output_dir):
    """
    The final step. The hard yard. Creation of the myelin map
    through division of two matrices. Calculation of this in the pre-computer
    era would have been a nightmare. Thankfully we don't live in those dark times...

    Inputs:
    corrected_t1 - Bias corrected + calibrated T1 image
    corrected_t2 - Bias corrected + calibrated T1 image

    Output:
    mm_image - The myelin map in subject space.
    """

    t1_hdr = nb.load(corrected_t1)
    t1_im = t1_hdr.get_data()

    t2_im = nb.load(corrected_t2).get_data()

    im_mm = t1_im / t2_im
    im_mm[np.isnan(im_mm)] = 0
    
    print('{} {}'.format(im_mm.min(), im_mm.max()))

    out_im_fn = os.path.join(output_dir, 'myelin_map_subj.nii.gz')
    nb.Nifti1Image(im_mm, affine = t1_hdr.affine, header = t1_hdr.header).to_filename(out_im_fn)

    return(out_im_fn)
