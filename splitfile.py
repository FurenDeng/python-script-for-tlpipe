import h5py as h5
import numpy as np
import ephem
from datetime import datetime, timedelta
timeformat = '%Y/%m/%d %H:%M:%S.%f'

def uneven_arr_split(array, section_size = 1):
    if type(section_size) is int:
        return np.array_split(array, section_size)
    if sum(section_size) != len(array):
        raise Exception('the section_size %d is not equal length of the array %d'%(sum(section_size), len(array)))

    new_arrs = []
    section_size = np.cumsum(np.append(0,section_size))
    for i in range(section_size.shape[0] - 1):
        new_arrs += [array[section_size[i]:section_size[i+1]]]
    return new_arr
def split(infile, outfiles, time_range = None, section_size = None):
    datain = h5.File(infile, 'r')
    if time_range is None:
        time = np.arange(datain['vis'].shape[0])
    else:
        time = np.arange(*time_range)
        if time.shape[0] > datain['vis'].shape[0]:
            raise Exception('time range is larger than the original time range!')
    if section_size is None:
        time_arrs = np.array_split(time, len(outfiles))
    elif sum(section_size) != len(time):
        raise Exception('the section_size %d is not equal length of time array %d'%(sum(section_size), len(time)))
    elif len(section_size) != len(outfiles):
        raise Exception('the section_size %d is not equal length of output file list %d'%(len(section_size), len(outfiles)))
    else:
        time_arrs = []
        section_size = np.cumsum(np.append(0,section_size))
        for i in range(section_size.shape[0] - 1):
            time_arrs += [list(np.arange(section_size[i],section_size[i+1]))]
        print(np.array(time_arrs).shape)
    print(time_arrs[0][0])
    print(time_arrs[-1][-1])
    for outfile, time_arr in zip(outfiles, time_arrs):
        print(time_arr[0])
        with h5.File(outfile,'w') as dataout:
            for attr in datain.attrs.keys():
                if attr == 'obstime':
                    obstime = datetime.strptime(datain.attrs['obstime'],timeformat)
                    dt = time_arr[0]*datain.attrs['inttime']
                    obstime += timedelta(seconds = dt)
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
infile = '3srcNP_20180101214415_20180101224415.hdf5'
outfile = ['3src_%i.hdf5'%i for i in range(6)]
split(infile, outfile, [300,3300],[500]*6)


