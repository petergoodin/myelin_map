from __future__ import division
import numpy as np
import nibabel as nb
from scipy import stats
from nibabel import processing as nbproc
import matplotlib.pylab as plt
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

        reg.inputs.num_threads = os.cpu_count()-1
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
    subj = os.path.split(output_dir)[-1]
    dir_files = os.listdir(output_dir)

    if fnmatch.filter(dir_files, '*mask_subj*'):
        print('Mask transforms already run. Not re-running for subj {}'.format(subj))

        subj_trans_masks = glob.glob(os.path.join(output_dir, '*mask_subj.nii.gz'))
        subj_trans_masks.sort(key = len)
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


        rigid.inputs.num_threads = os.cpu_count()-1
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
def bias_corr(images, brain_mask, output_dir):
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
            n4.inputs.n_iterations = [50,50,50,50]
            n4.inputs.save_bias = True
            n4.inputs.bias_image = os.path.join(output_dir, image_name + '_bias_field.nii.gz')
            n4.inputs.output_image = os.path.join(output_dir, image_name + '_bias_corr.nii.gz')
            n4.inputs.num_threads = os.cpu_count()-1
            n4.inputs.mask_image = brain_mask
            n4_results = n4.run()


            output_image = n4_results.outputs.get()['output_image']

            bias_output.append(output_image)

    return(bias_output)


def image_smooth(image_fn, fwhm = None, output_dir = None):

    subj = os.path.split(output_dir)[-1]

    print('Smoothing {} with kernel size: {}'.format(subj, fwhm))

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
                      brain_mni_mask = None,
                      output_dir = None):

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

    t1 = ['t1', t1_subj, t1_mni]
    t2 = ['t2', t2_subj, t2_mni]

    scan_data = [t1, t2]
    scan_fn_list = []

    for scan in scan_data:
        scan_type, scan_subj, scan_mni = scan
        print('***Calibrating {}***'.format(scan_type))

        #Extract stats

        Es = stats.mode(scan_subj[eye_subj_mask], axis = None)[0][0]
        Ms = stats.mode(scan_subj[temp_bone_subj_mask], axis = None)[0][0]

        print('\n{} mask values:\n{} eye = {} {} temp bone = {}'.format(subj, scan_type, Es, scan_type, Ms))

        Er = stats.mode(scan_mni[eye_mni_mask], axis = None)[0][0]
        Mr = stats.mode(scan_mni[temp_bone_mni_mask], axis = None)[0][0]

        print('\nMNI mask values:\n{} eye = {} {} temp bone = {}'.format(scan_type, Er, scan_type, Mr))


        #Shorten linear equation for easier troubleshooting
        eq_a = (Er - Mr) / (Es - Ms)
        print('Eq a val: {}'.format(eq_a))
        eq_b = ((Es * Mr) - (Er * Ms)) / (Es - Ms)
        print('Eq b val: {}'.format(eq_b))

        #Intensity correction
        scan_corr =  eq_a * scan_subj[brain_subj_mask] + eq_b

        print('{} bias corrected {}: {} {}'.format(subj, scan_type, scan_subj.min(), scan_subj.max()))

        print('{} calibrated {}: {} {}'.format(subj, scan_type, scan_corr.min(), scan_corr.max()))

        scan_corr_out = np.zeros_like(scan_subj)

        scan_corr_out[brain_subj_mask] = scan_corr
#         scan_corr_out = scan_corr

        if output_dir != None:
            scan_fn = os.path.join(output_dir, '{}_calibrated.nii.gz'.format(scan_type))
        else:
            scan_fn = '{}_calibrated.nii.gz'.format(scan_type)

        scan_fn_list.append(scan_fn)
        nb.Nifti1Image(scan_corr_out, affine = t1_subj_hdr.affine, header = t1_subj_hdr.header).to_filename(scan_fn)

    return(scan_fn_list)


###Myelin Map###
def create_mm_func(corrected_t1, corrected_t2, brain_mask, output_dir):
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

    t2_im = np.abs(nb.load(corrected_t2).get_data())

    brain_mask = nb.load(brain_mask).get_data().astype(bool)

    im_mm = np.zeros_like(t1_im)
    im_mm[brain_mask] = t1_im[brain_mask] / t2_im[brain_mask]
    im_mm[np.isnan(im_mm)] = 0
    im_mm[np.isinf(im_mm)] = 0
    print('MM Nans = {}'.format(np.isnan(im_mm).sum()))
    print('MM Inf = {}'.format(np.isinf(im_mm).sum()))

    im_mm[im_mm > 10] = 0

    print('{} myelin map values: {} {}'.format(subj, im_mm.min(), im_mm.max()))

    out_im_fn = os.path.join(output_dir, 'myelin_map_subj.nii.gz')
    nb.Nifti1Image(im_mm, affine = t1_hdr.affine, header = t1_hdr.header).to_filename(out_im_fn)

    return(out_im_fn)



def mm_minmax(mmap, mask, output_dir):
    """
    Takes the raw myelin map output from create_mm_func and minmax (min = 0,
    max = 1) standardises the voxels within the brain mask.
    """

    subj = os.path.split(output_dir)[-1]
    basename = os.path.split(mmap)[1].split('.')[0]
    outname = basename + '_minmax'

    print(subj, outname, output_dir)

    print('Running minmax standardisation on {}'.format(subj))

    mmap_hdr = nb.load(mmap)
    mmap_data = mmap_hdr.get_data()
    mask = nb.load(mask).get_data().astype(bool)

    mmap_flat = mmap_data[mask]
    mmap_flat_minmax = (mmap_flat - np.min(mmap_flat)) / (np.max(mmap_flat) - np.min(mmap_flat))
    print('{} minmax values: {} to {}'.format(subj, np.min(mmap_flat_minmax), np.max(mmap_flat_minmax)))
    im_mm = np.zeros_like(mmap_data)
    im_mm[mask] = mmap_flat_minmax

    out_im_fn = os.path.join(output_dir, outname + '.nii.gz')

    nb.Nifti1Image(im_mm, affine = mmap_hdr.affine, header = mmap_hdr.header).to_filename(out_im_fn)

    return(out_im_fn)
