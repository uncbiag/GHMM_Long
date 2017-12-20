"""
Run main script: Gaussian hidden Markov model (GHMM) pipeline
Usage: python ./GHMM_run_script.py ./data/ ./result/

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-11-18
"""

# import os
import time
import numpy as np
import glob
import nibabel as nib
from lmm import lmm_fun
from initial_b import initial_b_fun
from MRF_MAP import map_fun

"""
installed all the libraries above
"""


def run_script(input_dir, output_dir):

    """
    Run the commandline script for GHMM.

    :param
        input_dir (str): full path to the data folder
        output_dir (str): full path to the output folder
    """

    """+++++++++++++++++++++++++++++++++++"""
    print(""" Step 0. load dataset """)
    print("+++++++Read the image template file+++++++")
    template_name = input_dir + 'template.img'
    temp_img = nib.load(template_name)
    template = temp_img.get_data()
    num_vox = len(template[template == 1])
    print("The image size of all image data is " + str(temp.shape))
    print("+++++++Read the image data+++++++")
    img_folder_name = input_dir + "image"
    img_names = glob.glob("%s/*.img" % img_folder_name)
    num_img = len(img_names)
    img_data = np.zeros(shape=(num_img, num_vox))
    for ii in range(num_img):
        img_ii = nib.load(img_names[ii])
        imgdata_ii = img_ii.get_data()
        img_data[ii, :] = imgdata_ii[template == 1]
    print("The matrix dimension of image data is " + str(img_data.shape))
    print("+++++++Read the covariate data+++++++")
    design_data_file_name = input_dir + "design_data.txt"
    design_data = np.loadtxt(design_data_file_name)  # [the subject ID, visit time] (numeric)
    subject_id = np.unique(design_data[:, 0])
    num_sub = len(subject_id)
    print("The number of subjects is " + str(num_sub))
    print("+++++++Read the diagnostic information+++++++")
    dx_data_file_name = input_dir + "dx_data.txt"
    dx = np.loadtxt(dx_data_file_name)
    print("The number of normal subjects is " + str(num_sub[dx == 0]))
    print("The number of diseased subjects is " + str(num_sub[dx > 0]))

    """+++++++++++++++++++++++++++++++++++"""
    print("""Step 1. Preliminary settings""")
    start_1 = time.time()
    y_design = np.zeros(shape=(num_sub, num_vox))
    visit_data = design_data[:, 1]
    p = 2
    x_design = np.zeros(shape=(num_sub, p))
    for i in range(num_sub):
        sub_i_idx = design_data[:, 1] == subject_id[i]
        num_visit = len(sub_i_idx)
        y_design[i, :] = img_data[sub_i_idx[num_visit], :] - img_data[sub_i_idx[0], :]
        x_design[i, :] = [1, visit_data[sub_i_idx[num_visit], :] - visit_data[sub_i_idx[0], :]]
    print("+++++++Step 1.1: Set up linear mixed model on all voxels from normal subjects+++++++")
    [beta, s2] = lmm_fun(y_design, x_design, dx)[0]
    print("+++++++Step 1.2: Initial the diseased regions from patients (k-means)+++++++")
    nclasses = 2
    [b_0, mu] = initial_b_fun(y_design, x_design, dx, template, beta, nclasses)
    stop_1 = time.time()
    print("The cost time in Step 1 is %(t1) d" % {"t1": stop_1 - start_1})

    """+++++++++++++++++++++++++++++++++++"""
    print("""Step 2. Diseased region detection based on HMRF and EM""")
    start_2 = time.time()
    em_iter = 10
    map_iter = 10
    gamma = 0.2   # smooth parameter
    for out_it in range(em_iter):
        # update b via MAP algorithm
        b_0, mu = map_fun(b_0, x_design, y_design, dx, template, beta, mu, s2, gamma, nclasses, map_iter)
    b = b_0
    stop_2 = time.time()
    print("The cost time in Step 2 is %(t2) d" % {"t2": stop_2 - start_2})

    """+++++++++++++++++++++++++++++++++++"""
    print("""Step 3. Save the detected regions into image""")
    for ii in range(b.shape[0]):
        b_i = np.reshape(b[ii, :], template.shape)
        ana_img = nib.AnalyzeImage(b_i, np.eye(4))
        output_file_name = output_dir + 'b_%s.img' % ii
        nib.save(ana_img, output_file_name)

    if __name__ == '__main__':
        input_dir0 = sys.argv[1]
        output_dir0 = sys.argv[2]

        start_all = time.time()
        run_script(input_dir0, output_dir0)
        stop_all = time.time()
        delta_time_all = str(stop_all - start_all)
        print("The total elapsed time is " + delta_time_all)
