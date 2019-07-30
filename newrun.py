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

work_dir = '/public/furendeng/testspace' # the dir in which this script is running, the params file should be in this dir
result_dir = 'testdir' # the dir to save the result, should be the same as the pipe_outdir in the params file

result_dir = os.path.join(work_dir, result_dir)
ps_param_file = os.path.join(work_dir, 'ps_param_file.pipe')
ns_param_file = os.path.join(work_dir, 'ns_param_file.pipe')
abs_gain_file = os.path.join(result_dir, 'gain/cas_gain.hdf5')
param_data_file = os.path.join(work_dir, 'obs_data.hdf5') # the file read by process, will not change with time
data_prefix = '3src' # prefix of the datafile, use data_pre*suffix to find the datafile
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
# detect whether the absolute gain file exist
def Popen_save(cmd, outfile, errfile):

    process = sp.Popen(cmd + ' 1>%s 2>%s'%(outfile, errfile), shell = True, stdout = sp.PIPE, stderr = sp.PIPE)
    info = process.wait()
    print('Process finished!')
#    with open(outfile,'r') as fileout:
#        out = fileout.read()
    with open(errfile,'r') as fileerr:
        err = fileerr.read()
    return info, err
def process(outfile, errfile):
    # please do not resent me for such a long global variable list
    global abnormal_count, abnormal_max, ns_prop_file, abs_gain_file, preserve_transit_file, preserve_data_file, ns_arr_file, param_data_file, t, t0
    rebuild_prop = ''
    preserve_file = ''
    lost_point = ''
    # Too many batches of data are detected to be abnormal, the noise property file might need to be updated
    if abnormal_count >= abnormal_max:
        warnings.warn('%d batches of data are abnormal, rebuild the noise property file!'%abnormal_count)
        rebuild_prop = 'RebuildNoiseProperty_'
        abnormal_count = 0
        if os.path.exists(ns_prop_file):
            print('Removing %s'%ns_prop_file)
            sp.call('rm -rf %s'%ns_prop_file, shell = True)
        if os.path.exists(ns_arr_file):
            print('Removing %s'%ns_arr_file)
            sp.call('rm -rf %s'%ns_arr_file, shell = True)
    # if absolute cal has been done, just do the ns cal
    # if do not have enough noise points, wait for next file to come
    if path.isfile(abs_gain_file):
        cmd = 'mpiexec -n %d '%mpi_n + 'tlpipe %s'%ns_param_file
        info, err = Popen_save(cmd, outfile, errfile)
        if 'LostPointBegin' in err:
            warnings.warn('One lost point before the data is detected. There might be undetected lost point at the previous data.')
            lost_point = 'LostPointBegin_'
        if 'LostPointMiddle' in err:
            warnings.warn('One lost point in middle of this data is detected.')
            lost_point = 'LostPointMiddle_'
        if info:
            if 'NoiseNotEnough' in err:
                warnings.warn('For some reason, the absolute calibration has been done, but the noise property file was not established(or be deleted later). Wait for enough noise points!')
                return rebuild_prop + lost_point + 'NoiseNotEnough'
            elif 'NoNoisePoint' in err:
                warnings.warn('No noise point was found or the noise signal comes near the beginning or ending point! Skip this calibration')
                return rebuild_prop +  lost_point + 'NoNoisePoint'
            elif 'AbnormalPoints' in err:
                abnormal_count += 1
                warnings.warn('Too many abnormal points were found, skip this calibration!')
                return rebuild_prop +  lost_point + 'AbnormalPoints'
            elif 'DetectNoiseFailure' in err:
                warnings.warn('Fail to detect the property of noise! Use next batch of data!')
                if os.path.exists(ns_arr_file):
                    print('Removing %s'%ns_arr_file)
                    sp.call('rm -rf %s'%ns_arr_file, shell = True)
                return rebuild_prop +  lost_point + 'DetectNoiseFailure'
            elif 'IncontinuousData' in err:
                warnings.warn('Incontinuous data was detected when building up noise array! Skip this data and remove present noise array!')
                if os.path.exists(ns_arr_file):
                    print('Removing %s'%ns_arr_file)
                    sp.call('rm -rf %s'%ns_arr_file, shell = True)
                return rebuild_prop +  lost_point + 'IncontinuousData'
            elif 'OverlapData' in err:
                warnings.warn('Overlap data was detected when building up noise array! Skip this data and remove present noise array!')
                if os.path.exists(ns_arr_file):
                    print('Removing %s'%ns_arr_file)
                    sp.call('rm -rf %s'%ns_arr_file, shell = True)
                return rebuild_prop +  lost_point + 'OverlapData'
            else:
                print(err)
                warnings.warn('Some unexpected err was raised during noise calibration! Skip this calibration!')
                return rebuild_prop +  lost_point + 'UnexpectedError'
        print('Noise calibration succeed!')
        return rebuild_prop +  lost_point + 'ns_succeed'
    # if absolute cal has not been done, do it first
    # if preserve_transit_file and do not have enough noise points, save input file to preserve_data_file then concatenate the next input file and it.
    if preserve_transit_file:
        preserve_file = 'PreserveFile_'
        if not os.path.exists(preserve_data_file):
            pass
        else:
            with h5.File(preserve_data_file) as filep:
                with h5.File(param_data_file) as filed:
                    preserve_end_time = np.float128(filep.attrs['sec1970']) + filep.attrs['inttime'] * (filep['vis'].shape[0] - 1)
                    data_begin_time = np.float128(filed.attrs['sec1970'])
                    if np.around((data_begin_time - preserve_end_time)/filep.attrs['inttime']) == 1:
                        preserve_continuous_flag = True
                    else:
                        preserve_continuous_flag = False
            if preserve_continuous_flag:
                # if continuous, preserve. concatenate files and save the total file. else remove the preserved file.
                print('Concatenate %s and %s'%(preserve_data_file, param_data_file))
                with h5.File(preserve_data_file, 'r+') as fileout:
                    with h5.File(param_data_file, 'r') as filein:
                        new_vis = np.vstack([fileout['vis'], filein['vis']])
                        del fileout['vis']
                        fileout.create_dataset('vis', data = new_vis)
                print('Move concatenated file to %s'%(param_data_file))
                sp.call('mv %s %s'%(preserve_data_file, param_data_file) ,shell = True)
                t = time.time() - t0
                print('run time: %s'%datetime.timedelta(seconds = t))
            else:
                warnings.warn('Incontinuous or overlap data was detected when preserving data! Remove the preserved data!')
                print('Remove %s'%preserve_data_file)
                sp.call('rm -rf %s'%preserve_data_file, shell = True)


    cmd = 'mpiexec -n %d '%mpi_n + 'tlpipe %s'%ps_param_file
    info, err = Popen_save(cmd, outfile, errfile)
    bad_interp = 'NotEnoughPointToInterpolateError' in err
    if 'LostPointBegin' in err:
        warnings.warn('One lost point before the data is detected. There might be undetected lost point at the previous data.')
        lost_point = 'LostPointBegin_'
    if 'LostPointMiddle' in err:
        warnings.warn('One lost point in middle of this data is detected.')
        lost_point = 'LostPointMiddle_'
    return_txt = rebuild_prop + preserve_file + lost_point
    succeed = True # succeed in getting enough noise points if need to preserve data, always True if don't need to preserve
    if info:
        if 'NoTransit' in err:
            warnings.warn('Data contains no transition of point source! Skip it!')
            return_txt += 'NoTransit'
        else:
            if 'NoiseNotEnough' in err:
                warnings.warn('Data does not contain enough points to do noise calibration!')
                return_txt += 'NoiseNotEnough'
            elif 'NoNoisePoint' in err:
                warnings.warn('No noise point was found or the noise signal comes near the beginning or ending point! Skip this calibration')
                return_txt += 'NoNoisePoint'
            elif 'AbnormalPoints' in err:
                abnorma_count += 1
                warnings.warn('Too many abnormal points were found, may be there was lost point!')
                return_txt += 'AbnormalPoints'
            elif 'DetectNoiseFailure' in err:
                warnings.warn('Fail to detect the property of noise! Use next batch of data!')
                if os.path.exists(ns_arr_file):
                    print('Remove %s'%ns_arr_file)
                    sp.call('rm -f %s'%ns_arr_file, shell = True)
                return_txt += 'DetectNoiseFailure'
            elif 'IncontinuousData' in err:
                warnings.warn('Incontinuous data was detected when building up noise array! Skip this data and remove present noise array!')
                if os.path.exists(ns_arr_file):
                    print('Removing %s'%ns_arr_file)
                    sp.call('rm -rf %s'%ns_arr_file, shell = True)
                return_txt += 'IncontinuousData'
            elif 'OverlapData' in err:
                warnings.warn('Overlap data was detected when building up noise array! Skip this data and remove present noise array!')
                if os.path.exists(ns_arr_file):
                    print('Removing %s'%ns_arr_file)
                    sp.call('rm -rf %s'%ns_arr_file, shell = True)
                return_txt += 'OverlapData'
            elif bad_interp: # only preserve data if there is a transit, i.e. no other error was raised
                warnings.warn('More than half of the data was masked due to shortage of noise points for interpolation(need at least 4 to perform cubic spline)! The pointsource calibration may not be done due to too many masked points!')
                if os.path.exists(abs_gain_file):
                    print('Remove absolute gain file, and do absolute calibration again!')
                    sp.call('rm -rf %s'%abs_gain_file, shell = True)
                if not preserve_transit_file:
                    preserve_transit_file = True
                    preserve_file = 'PreserveFile_'
                    if not os.path.exists(preserve_data_file):
                        pass
                    else:
                        print('Remove old preserved data!')
                        sp.call('rm -rf %s'%preserve_data_file, shell = True)
                return_txt += 'NotEnoughPointToInterpolateError'
            else:
                return_txt += 'UnexpectedError'
                print(err)
                warnings.warn('Some unexpected err was raised during noise calibration! Skip this calibration!')
            if preserve_transit_file:
                print('Preserve data!')
                succeed = False
                if os.path.exists(ns_arr_file):
                    print('Remove %s'%ns_arr_file)
                    sp.call('rm -f %s'%ns_arr_file, shell = True)
                print('Preserve data to %s'%preserve_data_file)
                if not os.path.exists(os.path.dirname(preserve_data_file)):
                    os.makedirs(os.path.dirname(preserve_data_file))
                sp.call('mv %s %s'%(param_data_file, preserve_data_file), shell = True)
            else:
                warnings.warn('Do not have enough noise points, wait for enough noise points, so you may miss this transition!')
    else:
        print('Pointsource calibration succeed!')
        return_txt += 'ps_succeed'
    if succeed and preserve_transit_file:
        preserve_transit_file = False
        print('Copy %s to %s'%(param_data_file, preserve_data_file))
        sp.call('cp %s %s'%(param_data_file, preserve_data_file), shell = True)
    elif succeed:
        preserve_transit_file = False
    return return_txt

t0 =time.time()
input_file_list = []
input_file_status = []
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
try:
    while(1):
        t =time.time() - t0
        print('==================================')
        print('run time: %s'%datetime.timedelta(seconds = t))
        if len(input_file_list) < 5:
            print('Recent input files and status:')
            for file_name, file_status in zip(input_file_list, input_file_status):
                print(file_name + ': ' + file_status)
        else:
            print('Recent input files and status:')
            for file_name, file_status in zip(input_file_list[-5:], input_file_status[-5:]):
                print(file_name + ': ' + file_status)
        if (not time_limit is None) and t > time_limit:
            break
        files = glob.glob(data_prefix + '*' + data_suffix)
        print('current %s*%s file: '%(data_prefix, data_suffix))
        print(', '.join(files))
        old_files = set(input_file_list).intersection(files)
        if len(old_files) != 0:
            print('Remove expired files!')
        for old_file in old_files:
            print('Remove %s'%old_file)
            sp.call('rm -rf %s'%old_file, shell = True)
        if set(files).issubset(set(datafile)):
            time.sleep(time_interval)
            continue
        else:
            newfile = set(files) - set(datafile)
            newfile = list(newfile)
            newfile = sorted(newfile)
            if os.path.exists(ns_prop_file) and not preserve_transit_file:
                input_file_list += newfile
                input_file_status += ['DoesNotProcess']*(len(newfile) - 1)
                datafile += files
            else:
                input_file_list += [newfile[0]]
                datafile += [newfile[0]]
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
            t =time.time() - t0
            print('run time: %s'%datetime.timedelta(seconds = t))
            print('Start reading file!')
            sp.call('mv %s %s'%(newfile, param_data_file),shell = True)
            datafile = list(set(datafile))
            outfile = output_dir + '/' + newfile + '_output'
            errfile = output_dir + '/' + newfile + '_err'
            process_res = process(outfile, errfile)
            if 'LostPointBegin' in process_res and len(input_file_status) != 0:
                input_file_status[-1] += '_MayLostPointAtSomewhere'
            input_file_status += [process_res]
except KeyboardInterrupt:
    with open(output_dir + '/output_status.txt', 'w') as fileout:
        for file_name, file_status in zip(input_file_list, input_file_status):
            fileout.write(file_name + ': ' + file_status + '\n')





