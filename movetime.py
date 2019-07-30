import datetime
import h5py as h5
import numpy as np
import sys

'''
    used to change the obstime and sec1970 of a data file. move them backward(if time_interval is positive else forward).
    after delete one data point in series of data file, files after that point should be moved backward
'''

timeformat = '%Y/%m/%d %H:%M:%S.%f'
filename = raw_input('input filename:\n')
time_interval = raw_input('input the time interval that the obstime will be move backward:\n')
exec('time_interval = ' + time_interval)
with h5.File(filename, 'r+') as filein:
    obstime = filein.attrs['obstime']
    time = datetime.datetime.strptime(obstime, timeformat)
    time = time - datetime.timedelta(seconds = time_interval*filein.attrs['inttime'])
    filein.attrs['obstime'] = datetime.datetime.strftime(time, timeformat)
    filein.attrs['sec1970'] = filein.attrs['sec1970'] - time_interval*filein.attrs['inttime']
