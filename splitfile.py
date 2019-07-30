import h5py as h5
import numpy as np
import ephem
from datetime import datetime, timedelta
from uneven_split import uneven_arr_split 
timeformat = '%Y/%m/%d %H:%M:%S.%f'

def split(infile, outfiles, time_range = None, section_size = None):
    '''split observation data into several parts along time axis and store them into a series of small files
    Parameters:
    ----------
    infile: string. the file that is about to be split
    outfile: array. contains the small files that the result will be stored
    time_range: None, int or array. if int or array, specific time index range, only time index in np.arange(*time_range) will be stored into small files. default None, store all data.
    section_size: None or array. if array, split data into uneven parts. default None, split infile into len(outfiles) even parts as np.array_split.
    '''
    datain = h5.File(infile, 'r')
    if time_range is None:
        time = np.arange(datain['vis'].shape[0])
    elif not all(np.array(time_range) >= 0):
        raise Exception('elements in time_range should >=0!')
    else:
        time = np.arange(*time_range)
        if time[-1] > datain['vis'].shape[0]:
            raise Exception('time range is larger than the original time range!')
    if section_size is None:
        time_arrs = np.array_split(time, len(outfiles))
    elif sum(section_size) != len(time):
        raise Exception('the section_size %d is not equal length of time array %d'%(sum(section_size), len(time)))
    elif len(section_size) != len(outfiles):
        raise Exception('the section_size %d is not equal length of output file list %d'%(len(section_size), len(outfiles)))
    else:
        time_arrs = uneven_arr_split(time, section_size)
    print('start time index: %d'%time_arrs[0][0])
    print('end time index: %d'%time_arrs[-1][-1])
    for outfile, time_arr in zip(outfiles, time_arrs):
        print('process %s, from time index %d to %d'%(outfile, time_arr[0], time_arr[-1]))
        with h5.File(outfile,'w') as dataout:
            for attr in datain.attrs.keys():
                if attr == 'obstime':
                    obstime = datetime.strptime(datain.attrs['obstime'],timeformat)
                    dt = time_arr[0]*datain.attrs['inttime']
                    obstime += timedelta(seconds = dt)
                    print('obstime: %s'%obstime)
                    dataout.attrs['obstime'] = obstime.strftime(timeformat)
                elif attr == 'sec1970':
                    dataout.attrs['sec1970'] = datain.attrs['sec1970'] + time_arr[0]*datain.attrs['inttime']
                else:
                    dataout.attrs[attr] = datain.attrs[attr]
            for key in datain.keys():
                if key == 'vis':
                    dataout.create_dataset(key, data = datain[key][time_arr[0]:time_arr[-1] + 1])
                else:
                    dataout.create_dataset(key, data = datain[key][:])
                for attr in datain[key].attrs.keys():
                    dataout[key].attrs[attr] = datain[key].attrs[attr]
#infile = '3srcNP_20180101214415_20180101224415.hdf5'
infile = raw_input('input file:\n')
outfile = raw_input('output files(input python style command, such as [\'3src_%i.hdf5\'%i for i in range(6)]):\n')
outfile = 'outfile = ' + outfile
exec(outfile)
time_range = raw_input('input time range(int, None or array. input python stype command such as [300,3300]. default None):\n')
if len(time_range) == 0:
    time_range = None
else:
    exec('time_range = ' + time_range )
section_size = raw_input('input section_size(array or None. input python stype command such as [500]*6. default None):\n')
if len(section_size) == 0:
    section_size = None
else:
    exec('section_size = ' + section_size)

split(infile, outfile, time_range, section_size)


