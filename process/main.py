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
from error import *
from cali_params import params as pm


def Popen_save(cmd, outfile, errfile):
    process = sp.Popen(cmd + ' 1>%s 2>%s'%(outfile, errfile), shell = True, stdout = sp.PIPE, stderr = sp.PIPE)
    info = process.wait()
    print('Process finished!')
    with open(errfile,'r') as fileerr:
        err = fileerr.read()
    return info, err

pm.t0 =time.time()
pm.input_file_list = []
pm.input_file_status = []
if not os.path.exists(pm.output_dir):
    os.makedirs(pm.output_dir)
if not os.path.exists(os.path.dirname(pm.preserve_data_file)):
    os.makedirs(os.path.dirname(pm.preserve_data_file))
try:
    while(1):
        t =time.time() - pm.t0
        print('==================================')
        print('run time: %s'%datetime.timedelta(seconds = t))
        process_res = ''
        if pm.abnormal_count >= pm.abnormal_max:
            warnings.warn('%d batches of data are abnormal, rebuild the noise property file!'%pm.abnormal_count)
            process_res += 'RebuildNoiseProperty_'
            pm.abnormal_count = 0
            remove_file(pm.ns_prop_file)
            remove_file(pm.ns_arr_file)
        if len(pm.input_file_list) < 5:
            print('Recent input files and status:')
            for file_name, file_status in zip(pm.input_file_list, pm.input_file_status):
                print(file_name + ': ' + file_status)
        else:
            print('Recent input files and status:')
            for file_name, file_status in zip(pm.input_file_list[-5:], pm.input_file_status[-5:]):
                print(file_name + ': ' + file_status)
        if (not pm.time_limit is None) and t > pm.time_limit:
            break
        files = glob.glob(pm.data_prefix + '*' + pm.data_suffix)
        print('current %s*%s file: '%(pm.data_prefix, pm.data_suffix))
        print(', '.join(files))
        old_files = set(pm.input_file_list).intersection(files)
        if len(old_files) != 0:
            print('Remove expired files!')
        for old_file in old_files:
            print('Remove %s'%old_file)
            sp.call('rm -rf %s'%old_file, shell = True)
        if set(files).issubset(set(pm.datafile)):
            time.sleep(pm.time_interval)
            continue
        else:
            newfile = set(files) - set(pm.datafile)
            newfile = list(newfile)
            newfile = sorted(newfile)
            if os.path.exists(pm.ns_prop_file) and os.path.exists(pm.abs_gain_file) and not pm.preserve_transit_file:
                pm.input_file_list += newfile
                pm.input_file_status += ['DoesNotProcess']*(len(newfile) - 1)
                pm.datafile += files
            else:
                pm.input_file_list += [newfile[0]]
                pm.datafile += [newfile[0]]
                newfile = [newfile[0]]
            if len(newfile) > 1:
                warnings.warn('New files are more than one: %s\nOnly use the last one %s!'%(' ,'.join(newfile), newfile[-1]))
            else:
                print('Detected new file: %s'%', '.join(newfile))
            newfile = newfile[-1]
            while(1):
                try:
                    f = h5.File(newfile,'r')
                    f.close()
                    break
                except Exception as e:
                    print(e)
                    print('%s is not ready, wait for several seconds.'%newfile)
                    time.sleep(1.5)
            t =time.time() - pm.t0
            print('run time: %s'%datetime.timedelta(seconds = t))
            print('Start reading file!')
            sp.call('mv %s %s'%(newfile, pm.param_data_file),shell = True)
            pm.datafile = list(set(pm.datafile))
            outfile = pm.output_dir + '/' + newfile + '_output'
            errfile = pm.output_dir + '/' + newfile + '_err'
            if pm.preserve_transit_file:
                process_res += preserve_file(pm)
            if not os.path.exists(pm.abs_gain_file):
                cmd = 'mpiexec -n %d '%pm.mpi_n + 'tlpipe %s'%pm.ps_param_file
                info, err = Popen_save(cmd, outfile, errfile)
                process_res += error_process(err, pm, info, False)
            if os.path.exists(pm.abs_gain_file):
                cmd = 'mpiexec -n %d '%pm.mpi_n + 'tlpipe %s'%pm.ns_param_file
                info, err = Popen_save(cmd, outfile, errfile)
                process_res += error_process(err, pm, info, True)
            if 'LostPointBegin' in process_res and len(pm.input_file_status) != 0:
                pm.input_file_status[-1] += '_MayLostPointAtSomewhere'
            if 'AdditionalPointBegin' in process_res and len(pm.input_file_status) != 0:
                pm.input_file_status[-1] += '_MayAddedPointAtSomewhere'
            pm.input_file_status += [process_res]
except KeyboardInterrupt:
    with open(pm.output_dir + '/output_status.txt', 'w') as fileout:
        for file_name, file_status in zip(pm.input_file_list, pm.input_file_status):
            fileout.write(file_name + ': ' + file_status + '\n')


