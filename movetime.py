import datetime
import h5py as h5
import numpy as np
import sys

timeformat = '%Y/%m/%d %H:%M:%S.%f'
filename = sys.argv[1]
time_interval = int(float(sys.argv[2]))
with h5.File(filename, 'r+') as filein:
    obstime = filein.attrs['obstime']
    time = datetime.datetime.strptime(obstime, timeformat)
    time = time - datetime.timedelta(seconds = time_interval*filein.attrs['inttime'])
    filein.attrs['obstime'] = datetime.datetime.strftime(time, timeformat)
    filein.attrs['sec1970'] = filein.attrs['sec1970'] - time_interval*filein.attrs['inttime']
