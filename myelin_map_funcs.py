from __future__ import division
import numpy as np
import nibabel as nb
import scipy.stats as stats
from scipy.ndimage.morphology import binary_erosion as be
from statsmodels import robust
from nibabel import processing as nbproc
import matplotlib.pyplot as plt
import seaborn as sns
from nipype.interfaces import ants, dcm2nii
import fnmatch
import glob
import os


def dcm_convert(scan_dict, output_dir):

    subj = os.path.split(output_dir)[-1]
    dir_files = os.listdir(output_dir)


    if fnmatch.filter(dir_files, 't1.nii.gz') and fnmatch.filter(dir_files, 't2.nii.gz'):
        print('DCM to nifti conversion already run for {}. Not re-running.'.format(subj))

        #Get output from previous run, return in dictionary.
        dcm_out = glob.glob(os.path.join(output_dir, 't?.nii.gz'))
        t1_subj_nii = dcm_out[0]
        t2_subj_nii = dcm_out[1]

    else:
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
    #plt.show()

    fig = plt.figure(figsize = [15, 10]);
    plt.subplot(2,1,2);
    plt.title('Temporal Bone ' + stat)
    sns.distplot(t1[temp_mask], label = 'T1');
    ymin, ymax = fig.gca().axes.get_ybound()
    plt.vlines(x = t1_temp_stat, ymin = ymin, ymax = ymax)
    sns.distplot(t2[temp_mask], label = 'T2');
    plt.vlines(x = t2_temp_stat, ymin = ymin, ymax = ymax)
    plt.legend()
    #plt.show()

    plt.savefig('modes.png')

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

    subj = os.path.split(output_dir)[-1]

    if 'output_warped_image.nii.gz' in os.listdir(output_dir):

        print("Ants registration already run. Not re-running for subj {}. You're welcome.".format(subj))

        #Get output from previous run, return in dictionary.
        reg_trans = glob.glob(output_dir + '/reg_trans_*')
        reg_trans.append(glob.glob(output_dir + '/output_warped_image*')[0])
        reg_trans.sort()
        reg_labels = ['warped_image', 'trans_mat', 'composite_transform', 'inverse_composite_transform']
        reg_out_files = {k: reg_trans[n] for n, k in enumerate(reg_labels)}
        print(subj, reg_out_files)


    else:

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

        reg.inputs.num_threads = 2
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


    outputs:
    '''
    subj = os.path.split(output_dir)[-1]
    dir_files = os.listdir(output_dir)

    if fnmatch.filter(dir_files, '*mask_subj*'):
        print('Mask transforms already run. Not re-running for subj {}'.format(subj))

        subj_trans_masks = glob.glob(os.path.join(output_dir, '*mask_subj.nii.gz'))
        subj_trans_masks.sort() #Sure, this COULD have been a dict, but eh.
        print(subj, 'subj masks', subj_trans_masks)

    else:

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
            subj_trans_masks.sort() #Ditto

    return(subj_trans_masks)




#Register T2 to T1
def ants_rigid(fixed = None, moving = None, prefix = None, fixed_mask = None, moving_mask = None, output_dir = None):
    subj = os.path.split(output_dir)[-1]
    dir_files =  os.listdir(output_dir)

    if fnmatch.filter(dir_files, 'output_rigid*'):
        print("Ants rigid transformation already run. Not re-running for subj {}".format(subj))


        rigid_out_files = {'warped_image': os.path.join(output_dir, 'output_rigid_image.nii.gz')}
        print('Rigid out', rigid_out_files)

    else:
        print("Performing rigid registration of T1 and T2 images for subj {}".format(subj))

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


        rigid.inputs.num_threads = 1
        rigid.inputs.metric_weight = [1.0]
        rigid.inputs.winsorize_lower_quantile = 0.005
        rigid.inputs.winsorize_upper_quantile = 0.995
        rigid.inputs.verbose = True
        rigid.inputs.interpolation = 'NearestNeighbor' #Started with NN, gave good output.
        rigid_results = rigid.run()

        rigid_out_files = rigid_results.outputs.get()
        print('Rigid out', rigid_out_files)

    return(rigid_out_files)

#Myelin map to MNI
def subj2mni(moving = None, ref = None, transmat = None, output_dir = None):

    subj = os.path.split(output_dir)[-1]

    print('Warping {} to MNI space'.format(subj))

    subj2mni = ants.ApplyTransforms()
    subj2mni.inputs.dimension = 3
    subj2mni.inputs.input_image = moving
    subj2mni.inputs.reference_image = ref
    subj2mni.inputs.transforms = transmat


    if output_dir != None:
        subj2mni.inputs.output_image = os.path.join(output_dir, 'myelin_map_mni.nii.gz')
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

    subj = os.path.split(output_dir)[-1]
    dir_files = os.listdir(output_dir)

    if fnmatch.filter(dir_files, '*bias*'):
        print("Bias correction already run. Not re-running for subj {}".format(subj))

        bias_output = glob.glob(os.path.join(output_dir, '*_bias_corr.nii.gz'))
        bias_output.sort(key = len) #Assumes T2 image has been rigidly transformed to t1

    else:
        print("Running bias correction for subj {}".format(subj))
        bias_output = []

        for n, image in enumerate(images):
            image_file = os.path.split(image)[1]
            image_name = image_file.split('.')[0]
            print(image, image_name)


            n4  = ants.N4BiasFieldCorrection()
            n4.inputs.dimension = 3
            n4.inputs.input_image = image
            n4.inputs.bspline_fitting_distance = 300
            n4.inputs.shrink_factor = 2
            n4.inputs.n_iterations = [50,50,30,20]
            n4.inputs.save_bias = True
            n4.inputs.bias_image = os.path.join(output_dir, image_name + '_bias_field.nii.gz')
            n4.inputs.output_image = os.path.join(output_dir, image_name + '_bias_corr.nii.gz')
            n4.inputs.num_threads = 1
            n4_results = n4.run()


            output_image = n4_results.outputs.get()['output_image']

            bias_output.append(output_image)

    return(bias_output)


def image_smooth(image_fn, fwhm = None, output_dir = None):

    subj = os.path.split(output_dir)[-1]

    print('\nSmoothing {} with kernel size: {}'.format(subj, fwhm))

    im_name = os.path.split(image_fn)[-1].split('.')[0]

    im_hdr = nb.load(image_fn)

    smoothed = nbproc.smooth_image(img = im_hdr, fwhm = fwhm)

    output_fn = os.path.join(output_dir, im_name + '_smoothed_' + str(fwhm) + '_mm.nii.gz')

    nb.Nifti1Image(smoothed.get_data(), affine = im_hdr.affine, header = im_hdr.header).to_filename(output_fn)

    return(output_fn)



#Calibration stage
def image_calibration(t1_subj_bias_corr = None,
                      t2_subj_bias_corr = None,
                      t1_mni_bias_corr = None,
                      t2_mni_bias_corr = None,
                      eye_subj_mask = None,
                      temp_bone_subj_mask = None,
                      brain_subj_mask = None,
                      eye_mni_mask = None,
                      temp_bone_mni_mask = None,
                      brain_mni_mask = None, output_dir = None):

    subj = os.path.split(output_dir)[-1]


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
    t1_mni_eye = t1_mni[eye_mni_mask]
    t1_mni_eye_stat = stats.mode(t1_mni_eye)[0][0]
    t1_mni_temp = t1_mni[temp_bone_mni_mask]
    t1_mni_temp_stat = stats.mode(t1_mni_temp)[0][0]
    t2_mni_eye = t2_mni[eye_mni_mask]
    t2_mni_eye_stat = stats.mode(t2_mni_eye)[0][0]
    t2_mni_temp =t2_mni[temp_bone_mni_mask]
    t2_mni_temp_stat = stats.mode(t2_mni_temp)[0][0]

    print('\nMNI mask values:\nT1 eye = {} T1 temp bone = {}\nT2 eye = {} T2 temp bone = {}\n'.format(t1_mni_eye_stat, t1_mni_temp_stat, t2_mni_eye_stat, t2_mni_temp_stat))

    t1_subj_eye = t1_subj[eye_subj_mask]
    t1_subj_eye_stat = stats.mode(t1_subj_eye)[0][0]
    t1_subj_temp = t1_subj[temp_bone_subj_mask]
    t1_subj_temp_stat = stats.mode(t1_subj_temp)[0][0]
    t2_subj_eye = t2_subj[eye_subj_mask]
    t2_subj_eye_stat = stats.mode(t2_subj_eye)[0][0]
    t2_subj_temp =t2_subj[temp_bone_subj_mask]
    t2_subj_temp_stat = stats.mode(t2_subj_temp)[0][0]

    print('\n{} mask values:\nT1 eye = {} T1 temp bone = {}\nT2 eye = {} T2 temp bone = {}'.format(subj, t1_subj_eye_stat, t1_subj_temp_stat, t2_subj_eye_stat, t2_subj_temp_stat))

    with open(os.path.join(output_dir, subj + '_mask_values.txt'), 'w') as text_file:
        text_file.write('{} mask values:\nT1 eye = {} T1 temp bone = {}\nT2 eye = {} T2 temp bone = {}'.format(subj, t1_subj_eye_stat, t1_subj_temp_stat, t2_subj_eye_stat, t2_subj_temp_stat))

    #FOR FUTURE: SPLIT INTO 2 FUNCTIONS


    #Shorten linear equation for easier troubleshooting
    t1_a = (t1_mni_temp_stat - t1_mni_eye_stat) / (t1_subj_temp_stat - t1_subj_eye_stat)
    t1_b = ((t1_subj_temp_stat * t1_mni_eye_stat) - (t1_mni_temp_stat * t1_subj_eye_stat)) / (t1_subj_temp_stat - t1_subj_eye_stat)

    t2_a = (t2_mni_temp_stat - t2_mni_eye_stat) / (t2_subj_temp_stat - t2_subj_eye_stat)
    t2_b = ((t2_subj_temp_stat * t2_mni_eye_stat) - (t2_mni_temp_stat * t2_subj_eye_stat)) / (t2_subj_temp_stat - t2_subj_eye_stat)


    #Intensity correction
    t1_corr = (t1_a * t1_subj[brain_subj_mask]) + t1_b
    t2_corr = (t2_a * t2_subj[brain_subj_mask]) + t2_b

    print('\n{} bias corrected T1: {} {}'.format(subj, t1_subj.min(), t1_subj.max()))
    print('{} bias corrected T2: {} {}'.format(subj, t1_subj.min(), t1_subj.max()))

    print('{} calibrated T1: {} {}'.format(subj, t1_corr.min(), t1_corr.max()))
    print('{} calibrated T2:{} {}'.format(subj, t2_corr.min(), t2_corr.max()))

    t1_corr_out = np.zeros_like(t1_subj)
    t2_corr_out = np.zeros_like(t2_subj)

    t1_corr_out[brain_subj_mask] = t1_corr
    t2_corr_out[brain_subj_mask] = t2_corr

    if output_dir != None:
        t1_fn = os.path.join(output_dir, 't1_calibrated.nii.gz')
        t2_fn = os.path.join(output_dir, 't2_calibrated.nii.gz')
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

    subj = os.path.split(output_dir)[-1]

    t1_hdr = nb.load(corrected_t1)
    t1_im = t1_hdr.get_data()
    t1_mask = t1_im > 0

    t2_im = nb.load(corrected_t2).get_data()
    t2_mask = t2_im > 0
    # n_iters = 1
    # print('\nRemoving {} voxels at zero boundaries'.format(1 + n_iters))
    # t2_mask = be(t2_mask, iterations = n_iters)

    # dif_mask = (t1_mask.astype(int) - t2_mask.astype(int)).astype(bool)
    # dif_mask = t1_mask.astype(int) - t2_mask.astype(int)

    mm = t1_im[t2_mask] / t2_im[t2_mask]
    im_mm = np.zeros_like(t1_im)
    im_mm[t2_mask] = mm

    im_mm[np.isnan(im_mm)] = 0
    im_mm[im_mm < 0] = 0
    im_mm[im_mm >= 6] = 0 #Note, 6 is a value that works well to filter edge voxels with massive values.
    # im_mm = im_mm * (-1 * dif_mask)
    # im_mm[im_mm[dif_mask] == 1] = 0

    print('\n{} myelin map values: \nmin = {}\nmax = {}\nmean = {} ({})\nmedian = {}'.format(subj, im_mm.min(),
                                                                                                im_mm.max(),
                                                                                                im_mm[im_mm > 0].mean(),
                                                                                                np.std(im_mm[im_mm > 0]),
                                                                                                np.median(im_mm[im_mm > 0])))

    out_im_fn = os.path.join(output_dir, 'myelin_map_subj.nii.gz')
    nb.Nifti1Image(im_mm, affine = t1_hdr.affine, header = t1_hdr.header).to_filename(out_im_fn)

    return(out_im_fn)


def mm_minmax(mmap, mask, output_dir):
    """
    Takes the raw myelin map output from create_mm_func and performs conversion to percentiles
    """

    subj = os.path.split(output_dir)[-1]
    basename = os.path.split(mmap)[1].split('.')[0]
    outname = basename + '_minmax'

    print(subj, outname, output_dir)

    print('\nRunning minmax standardisation on {}'.format(subj))

    mmap_hdr = nb.load(mmap)
    mmap_data = mmap_hdr.get_data()
    mask = nb.load(mask).get_data().astype(bool)

    max_val = mmap_data.max()

    im_mm = mmap_data / max_val

    # mmap_flat = mmap_data[mask]
    # mmap_flat_minmax = (mmap_flat - np.min(mmap_flat)) / (np.max(mmap_flat) - np.min(mmap_flat))
    # print('{} minmax values: {} to {}'.format(subj, np.min(mmap_flat_minmax), np.max(mmap_flat_minmax)))
    # im_mm = np.zeros_like(mmap_data)
    # im_mm[mask] = mmap_flat_minmax

    out_im_fn = os.path.join(output_dir, outname + '.nii.gz')

    nb.Nifti1Image(im_mm, affine = mmap_hdr.affine, header = mmap_hdr.header).to_filename(out_im_fn)

    return(out_im_fn)



###FUNCTION FOR PARALLEL PROCESSING###

def myelin_map_proc(subj, n_cores, raw_dir, output_dir, patterns, n_scans, dcm_suffix, fwhm_list):
    """
    Wrapper function that is needed for parallel processing but can also be used for individual subjects.

    Inputs:
    subj - name of participant to be processed (str)
    n_cores - number of cores to be used for parallel processing (int).
    note: for single participants only a single core is used).
    raw_dir - path to root directory which houses subj/dicoms (str).
    output_dir - path to write output to (note: output_dir is the root. Individual participant directories will be created during the process (str).
    patterns - strings for how to recognise T1 and T2 dicom directories with the participant directory (dict).
    n_scans - Number of expected dicom files in t1 and t2 directories (dict).
    dcm_suffix - Strings of file endings for how to recognise dicom files (str).
    fwhm_list - Values for size of smoothing kernel (in mm) (list).
    Note: Can be single value or multiple.

    Output:
    Everything.
    """

    print('Processing data for subj {}'.format(subj))

    try:
        os.mkdir(output_dir)
    except:
        print('Directory {} exists. Not creating'.format(output_dir))

    out_subj_dir = os.path.join(output_dir, subj)
    print()

    try:
        os.mkdir(out_subj_dir)
    except:
        print('Directory {} already exists. Not creating.'.format(out_subj_dir))

    print("\nWorking dir: {}".format(out_subj_dir))

    ##Read in masks
    t1_im_mni_fn = os.path.join('.', 'resources','mni_t1_template.nii.gz')
    t2_im_mni_fn = os.path.join('.', 'resources','mni_t2_template.nii.gz')


    eye_mni_mask_fn = os.path.join('.', 'resources','mni_eye_mask.nii.gz')
    temp_bone_mni_mask_fn = os.path.join('.', 'resources','mni_temp_bone_mask.nii.gz')
    brain_mni_mask_fn = os.path.join('.', 'resources','mni_brain_mask.nii.gz')



    scan_dict = {'t1': glob.glob(os.path.join(raw_dir, subj, patterns[0], '*{}'.format(dcm_suffix))),
                 't2': glob.glob(os.path.join(raw_dir, subj, patterns[1], '*{}'.format(dcm_suffix)))
                 }

    #Count number of scans for t1 and t2, print and write to file.
    im_count = {k: len(list(scan_dict.values())[n]) for n, k in enumerate(scan_dict)}
    print(subj, im_count)

    with open(os.path.join(out_subj_dir, subj + '_input_files_n.txt'), 'w') as text_file:
        text_file.write('{} - {}'.format(subj, im_count))


    #Generate error if data not matching criteria
    if len(scan_dict['t1']) < n_scans['t1']:
        with open(os.path.join(out_subj_dir, subj + '_t1_error.txt'), 'w') as text_file:
            text_file.write('Number of T1 scans < {}}: {}'.format(n_scans['t1'], len(scan_dict['t1'])))
        raise ValueError('Error: Number of DICOMS in T1 directory less than expected for subj {}\n# of Dicoms = {}'.format(subj, len(scan_dict['t1'])))

    if len(scan_dict['t2']) < n_scans['t2']:
        with open(os.path.join(out_subj_dir, subj + '_t2_error.txt'), 'w') as text_file:
            text_file.write('Number of T2 scans < {}: {}'.format(n_scans['t2'], len(scan_dict['t2'])))
        raise ValueError('Error: Number of DICOMS in T2 directory less than expected for subj {}\n# of Dicoms = {}'.format(subj, len(scan_dict['t2'])))


    #Start pipeline
    t1_subj_nii, t2_subj_nii = dcm_convert(scan_dict, out_subj_dir)

    reg_output = ants_reg(fixed = t1_subj_nii, moving = t1_im_mni_fn, output_dir = out_subj_dir)

    mask_list = [eye_mni_mask_fn, temp_bone_mni_mask_fn, brain_mni_mask_fn]

    brain_subj_mask, eye_subj_mask, temp_bone_subj_mask,  = mask_transform(mask_list = mask_list,
                                                                         ref = t1_subj_nii,
                                                                         transmat = reg_output['composite_transform'],
                                                                         output_dir = out_subj_dir)

#    plot_ants_warp(fixed = t1_subj_nii, moving = brain_subj_mask, nslices = 10, output_name = os.path.join(out_subj_dir, 'brain_mask'))

    rigid_output = ants_rigid(fixed = t1_subj_nii, moving = t2_subj_nii, output_dir = out_subj_dir)

    t1_bias, t2_bias = bias_corr([t1_subj_nii, rigid_output['warped_image']], output_dir = out_subj_dir)

    #Calibration - TO DO: Split function into mode calculation + linear correction

    print('Brain = {}\nEye = {}\nTemp = {}'.format(brain_subj_mask, eye_subj_mask, temp_bone_subj_mask))

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
                                                           output_dir = out_subj_dir)

    #Calculate myelin maps
    myelin_map_subj = create_mm_func(t1_cal, t2_cal, output_dir = out_subj_dir)

    #Warp myelin maps to MNI space
    subj2mni_output, subj2mni_im = subj2mni(moving = myelin_map_subj, ref = t1_im_mni_fn, transmat = reg_output['inverse_composite_transform'], output_dir = out_subj_dir)

    #Smooth

    if len(fwhm_list) > 1:
        for im in [myelin_map_subj, subj2mni_im]:
            for fwhm in fwhm_list:
                smoothed = image_smooth(image_fn = im, fwhm = fwhm, output_dir = out_subj_dir)
    else:
        for im in [myelin_map_subj, subj2mni_im]:
            smoothed = image_smooth(image_fn = im, fwhm = fwhm_list, output_dir = out_subj_dir)


    #minmax
    mmap_list = glob.glob(os.path.join(out_subj_dir, 'myelin_map*'))
    mmap_list = [im for im in mmap_list if 'minmax' not in im]
    print(mmap_list)

    for im in mmap_list:
        if 'mni' in im:
            mm_minmax(mmap = im, mask = brain_mni_mask_fn, output_dir = out_subj_dir)
        else:
            mm_minmax(mmap = im, mask = brain_subj_mask, output_dir = out_subj_dir)
