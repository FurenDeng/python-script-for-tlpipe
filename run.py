import numpy as np
import h5py as h5
import os
import sys
import time
from os import path
import subprocess as sp
import glob
import warnings

ps_param_file = 'ps_param_file.pipe'
ns_param_file = 'ns_param_file.pipe'
abs_gain_file = '/public/furendeng/testspace/testdir/gain/cas_gain.hdf5'
param_data_file = 'obs_data.hdf5' # the file read by process, will not change with time
data_pre = '3src' # prefix of the datafile, use data_pre*.hdf5 to find the datafile
time_interval = 2 # in second
mpi_n = 16
time_limit = None # in second, stop when process time is larger than it
preserve_transit_file = True # preserve transit file if don't have enough points to do the noise calibration, which will lead to removal of ns_arr_file and merge of several data file
ns_arr_file = '/public/furendeng/testspace/testdir/ns_cal/ns_arr_file.npz'
preserve_data_file = '/public/furendeng/testspace/testdir/preserve/preserve.hdf5'
datafile = [param_data_file, os.path.basename(abs_gain_file), os.path.basename(preserve_data_file)] # new file added to this list should be the observed visibility, and the list will change with time. New file should be mv into param_data_file

# detect whether the absolute gain file exist
def process(abs_gain_file, ps_param_file, ns_param_file, mpi_n, preserve_transit_file, preserve_data_file, param_data_file):
    # if absolute cal has been done, just do the ns cal
    # if do not have enough noise points, wait for next file to come
    if path.isfile(abs_gain_file):
        cmd = 'mpiexec -n %d '%mpi_n + 'tlpipe %s'%ns_param_file
        p = sp.Popen(cmd, shell = True, stderr = sp.PIPE)
        info = p.wait()
        out, err = p.communicate()
        if info:
            if 'NoiseNotEnough' in err:
                print(err)
            else:
                raise Exception(err)
        else:
            print(err)
        return True
    # if absolute cal has not been done, do it first
    # if preserve_transit_file and do not have enough noise points, save input file to preserve_data_file then concatenate the next input file and it.
    if preserve_transit_file:
        if not os.path.exists(preserve_data_file):
            pass
        else:
            # concatenate files and save the total file
#            preserved_file = os.path.dirname(param_data_file) + '/' + os.path.basename(preserve_data_file)
#            print('move %s to %s'%(preserve_data_file, preserved_file))
#            sp.call('mv %s %s'%(preserve_data_file, preserved_file), shell = True)
            print('Concatenate %s and %s'%(preserve_data_file, param_data_file))
            with h5.File(preserve_data_file, 'r+') as fileout:
                with h5.File(param_data_file, 'r') as filein:
                    new_vis = np.vstack([fileout['vis'][:], filein['vis'][:]]).copy()
                    del fileout['vis']
                    fileout.create_dataset('vis', data = new_vis)
            print('Move concatenated file to %s'%(param_data_file))
            sp.call('mv %s %s'%(preserve_data_file, param_data_file) ,shell = True)


    cmd = 'mpiexec -n %d '%mpi_n + 'tlpipe %s'%ps_param_file
    p = sp.Popen(cmd, shell = True, stderr = sp.PIPE)
    info = p.wait()
    out, err = p.communicate()
    succeed = True # succeed in getting enough noise points if need to preserve data, always True if don't need to preserve
    if info:
        if 'NoTransit' in err or 'NoiseNotEnough' in err:
            print(err)
        else:
            raise Exception(err)
        if 'NoiseNotEnough' in err:
            if preserve_transit_file:
                succeed = False
                print('remove %s'%ns_arr_file)
                sp.call('rm -f %s'%ns_arr_file, shell = True)
                print('Preserve data to %s'%preserve_data_file)
                if not os.path.exists(os.path.dirname(preserve_data_file)):
                    os.makedirs(os.path.dirname(preserve_data_file))
                sp.call('mv %s %s'%(param_data_file, preserve_data_file), shell = True)
            else:
                warnings.warn('Do not have enough noise points, wait for enough noise points, so you will miss this transition!')
    else:
        print(err)
    if succeed and preserve_transit_file:
        preserve_transit_file = False
        sp.call('cp %s %s'%(param_data_file, preserve_data_file), shell = True)
    elif succeed:
        preserve_transit_file = False
    return False

t0 =time.time()
while(1):
    t =time.time() - t0
    print('run time: %.2f'%t)
    print('datafile list: ')
    print(datafile)
    if (not time_limit is None) and t > time_limit:
        break
    files = glob.glob(data_pre + '*.hdf5')
    print('current %s*.hdf5 file: '%data_pre)
    print(files)
    if set(files).issubset(set(datafile)):
        time.sleep(time_interval)
        continue
    else:
        newfile = set(files) - set(datafile)
        datafile += files
        if len(newfile) > 1:
            raise Exception('new file is more than one!')
        newfile = list(newfile)[0]
        print('Detected new file: %s'%newfile)
        while(1):
            try:
                f = h5.File(newfile,'r')
                f.close()
                break
            except Exception as e:
                print(e)
                print('%s is not ready, wait for several seconds.'%newfile)
                time.sleep(1.5)
        t =time.time() - t0
        print('run time: %.2f'%t)
        print('Start reading file!')
        sp.call('mv %s %s'%(newfile, param_data_file),shell = True)
        datafile = list(set(datafile))
        process(abs_gain_file, ps_param_file, ns_param_file, mpi_n, preserve_transit_file, preserve_data_file, param_data_file)






