import numpy as np
import h5py as h5
import os
import sys
import time
from os import path
import subprocess as sp
import glob
import warnings
import datetime
import sys
class params:
    work_dir = '/public/furendeng/testspace' # the dir in which this script is running, the params file should be in this dir
    result_dir = 'testdir' # the dir to save the result, should be the same as the pipe_outdir in the params file

    result_dir = os.path.join(work_dir, result_dir)
    ps_param_file = os.path.join(work_dir, 'ps_param_file.pipe')
    ns_param_file = os.path.join(work_dir, 'ns_param_file.pipe')
    abs_gain_file = os.path.join(result_dir, 'gain/cas_gain.hdf5')
    param_data_file = os.path.join(work_dir, 'obs_data.hdf5') # the file read by process, will not change with time
    data_prefix = 'Cas' # prefix of the datafile, use data_pre*suffix to find the datafile
    data_suffix = '.hdf5'# suffix of the datafile, use data_pre*suffix to find the datafile
    time_interval = 2 # in second
    mpi_n = 16
    time_limit = None # in second, stop when process time is larger than it
    preserve_transit_file =  True # preserve transit file if don't have enough points to do the noise calibration, which will lead to removal of ns_arr_file and merge of several data file
    ns_arr_file = os.path.join(result_dir, 'ns_cal/ns_arr_file.npz') # the saved noise array, will be used to do the noise cal if the noise points in one data is not enough
    ns_prop_file = os.path.join(result_dir, 'ns_cal/ns_prop_file.npz') # the property of noise, will be used to predict the noise points
    preserve_data_file = os.path.join(result_dir, 'preserve/preserve.hdf5') # save the preserved file to here
    output_dir = os.path.join(result_dir, 'output') # to save the stdout and stderr
    datafile = [param_data_file, os.path.basename(abs_gain_file), os.path.basename(preserve_data_file)] # new file added to this list should be the observed visibility, and the list will change with time. New file should be mv into param_data_file
    abnormal_max = 3 # if abnormal_count >= abnormal_max, rebuild the noise property file
    abnormal_count = 0 # do not chnage, count the abnormal data
