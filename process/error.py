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

exception_dict = {
# from detect_ns.py
    'IncontinuousData':#=================================================================
    ['Incontinuous data was detected when building up noise array! Skip this data and remove present noise array!','IncontinuousData_'],
    'OverlapData':#======================================================================
    ['Overlap data was detected when building up noise array! Skip this data and remove present noise array!','OverlapData_'],
    'NoiseNotEnough':#===================================================================
    ['Data does not contain enough points to do noise calibration! Wait for more!','NoiseNotEnough_'],
    'AbnormalPoints':#===================================================================
    ['Too many abnormal points were found, skip this calibration!','AbnormalPoints_'],
    'NoNoisePoint':#=====================================================================
    ['No noise point was found or the noise signal comes near the beginning or ending point! Skip this calibration','NoNoisePoint_'],
    'DetectNoiseFailure':#===============================================================
    ['Fail to detect the property of noise! Use next batch of data!','DetectNoiseFailure_'],
# from ns_cal.py
    'NoPsGainFile':#=====================================================================
    ['No absolute gain file, do the ps calibration first!','NoPsGainFile_'],
    'TransitGainNotRecorded':#===========================================================
    ['The transit is not in time range and no transit normalization gain was recorded!','TransitGainNotRecorded_'],
    'BeforeStableTime':#=================================================================
    ['The beginning time point of this data is earlier than the stable time. Abort the noise calibration!','BeforeStableTime_'],
    'AllMasked':#========================================================================
    ['All values are masked when calculating phase or amplitude!','AllMasked_'],
# from ps_cal.py
    'NoTransit':#========================================================================
    ['Data does not contain local transit time.','NoTransit_'],
    'NotEnoughPointToInterpolateError':#=================================================
    ['More than 80% of the data was masked due to shortage of noise points for interpolation(need at least 4 to perform cubic spline)! The pointsource calibration may not be done due to too many masked points!', 'NotEnoughPointToInterpolateError_'],
    }
warning_dict = {
# from detect_ns.py
    'LostPointBegin':
    ['One lost point before the data is detected. There might be undetected lost point at the previous data.','LostPointBegin_',True],

    'LostPointMiddle':
    ['One lost point in middle of this data is detected.','LostPointMiddle_',True],

    'AdditionalPointBegin':
    ['One additional point before the data is detected. There might be undetected additional point at the previous data.','AdditionalPointBegin_',True],

    'AdditionalPointMiddle':
    ['One additional point in middle of this data is detected.','AdditionalPointMiddle_',True],

    'DetectedLostAddedAbnormal':
    ['Abnormal additional or lost points are detected for some channel.','DetectedLostAddedAbnormal_',False],

    'ChangeReferenceTime':
    ['Move the reference time one index earlier to compensate!','ChangeReferenceTime_',True],

# from ns_cal.py
    'NotEnoughPointToInterpolateWarning':
    ['More than half of the data was masked due to shortage of noise points for interpolation(need at least 4 to perform cubic spline)! The pointsource calibration may not be done due to too many masked points!','NotEnoughPointToInterpolateWarning_',False],

    'TransitMasked':
    ['The transit point has been masked for some frequencies baselines when calculating amplitude! Maybe the noise points are too sparse!','TransitMasked',False],

    }
def remove_file(filename):
    if os.path.exists(filename):
        print('Removing %s'%filename)
        sp.call('rm -rf %s'%filename, shell = True)

def preserve_file(pm):
    # preserve param_data_file
    # remove the ns_arr_file to prevent overlap
    # concatenate it and preserve_data_file
    # if overlap or incontinuity was detected, remove the preserve_data_file.
    if not os.path.exists(pm.preserve_data_file):
        return 'StartPreserve_'
    else:
        if os.path.exists(pm.ns_arr_file):
            sp.call('rm -rf %s'%pm.ns_arr_file, shell=True)
        with h5.File(pm.preserve_data_file) as filep:
            with h5.File(pm.param_data_file) as filed:
                preserve_end_time = np.float128(filep.attrs['sec1970']) + filep.attrs['inttime'] * (filep['vis'].shape[0] - 1)
                data_begin_time = np.float128(filed.attrs['sec1970'])
                if np.around((data_begin_time - preserve_end_time)/filep.attrs['inttime']) == 1:
                    preserve_continuous_flag = True
                else:
                    preserve_continuous_flag = False
        if preserve_continuous_flag:
            # if continuous, preserve. concatenate files and save the total file. else remove the preserved file.
            print('Concatenate %s and %s'%(pm.preserve_data_file, pm.param_data_file))
            with h5.File(pm.preserve_data_file, 'r+') as fileout:
                with h5.File(pm.param_data_file, 'r') as filein:
                    new_vis = np.vstack([fileout['vis'], filein['vis']])
                    del fileout['vis']
                    fileout.create_dataset('vis', data = new_vis)
            print('Move concatenated file to %s'%(pm.param_data_file))
            sp.call('mv %s %s'%(pm.preserve_data_file, pm.param_data_file) ,shell = True)
            t = time.time() - pm.t0
            print('run time: %s'%datetime.timedelta(seconds = t))
            return 'PreserveFile_'
        else:
            warnings.warn('Incontinuous or overlap data was detected when preserving data! Remove the preserved data!')
            print('Remove %s'%pm.preserve_data_file)
            sp.call('rm -rf %s'%pm.preserve_data_file, shell = True)
            return 'FailedPreserve_'

def error_process(err, pm, info, after_ps = False):
    errors = []
    warns = []
    status = ''
#    preserve_transit_file = pm.preserve_transit_file and not after_ps
    for warn_name, (message, sta, record) in warning_dict.items():
        if warn_name in err and record:
            print(message)
            warns += [warn_name]
            status += sta
    for error_name, (message, sta) in exception_dict.items():
        if error_name in err:
            print(message)
            errors += [error_name]
            status += sta
    if len(errors)==0:
        if info:
            status += 'UnexpectedError_'
            print('Some unexpected err was raised during noise calibration! Skip this calibration!')
        else:
            if pm.preserve_transit_file:
                print('Calibration succeed. Stop preservation.')
                pm.preserve_transit_file = False
                print('Copy %s to %s'%(pm.param_data_file, pm.preserve_data_file))
                sp.call('cp %s %s'%(pm.param_data_file, pm.preserve_data_file), shell = True)
            # no error, warnings all do not need to deal with
            if after_ps:
                status += 'NsSucceed'
            else:
                status += 'PsSucceed'
            return status
    if 'NoTransit' in errors and pm.preserve_transit_file:
        print('Noise property file has been built and no transit. Stop preservation.')
        # no transit and detect_ns works, do not need to preserve
        print('Copy %s to %s'%(pm.param_data_file, pm.preserve_data_file))
        sp.call('cp %s %s'%(pm.param_data_file, pm.preserve_data_file), shell = True)
        pm.preserve_transit_file = False
    if 'AllMasked' in errors:
        print('Preserve data to do noise calibration at transit point!')
        pm.preserve_transit_file = True
    if pm.preserve_transit_file:
        print('Move %s to %s'%(pm.param_data_file,pm.preserve_data_file))
        sp.call('mv %s %s'%(pm.param_data_file,pm.preserve_data_file), shell=True)
    if 'AbnormalPoints' in errors:
        pm.abnormal_count += 1
    if 'OverlapData' in errors or 'IncontinuousData' in errors or 'DetectNoiseFailure' in errors:
        remove_file(pm.ns_arr_file)
    if after_ps:
        status += 'NsFail'
    else:
        status += 'PsFail'
    return status




























