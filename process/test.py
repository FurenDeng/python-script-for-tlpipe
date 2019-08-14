import numpy as np
import h5py as h5
from glob import glob
import matplotlib.pyplot as plt

file1 = []
file2 = []

for i in range(6,180,1):
    if i <10:
        file1 += ['save_result/Cas_0%d.hdf5_output'%i]
        file2 += ['../../data/long_test/Cas_0%d.hdf5'%i]
    else:
        file1 += ['save_result/Cas_%d.hdf5_output'%i]
        file2 += ['../../data/long_test/Cas_%d.hdf5'%i]

bln = 150
ifreq = 15
freq = None
for ii, filename in enumerate(file1):
    try:
        with h5.File(filename + '/ns_cal/gain.hdf5', 'r') as filein:
            print(filename)
            if freq is None:
                print(filein['bl_order'][bln])
                freq = filein['freq'][ifreq]
                print(freq)
            time_inds = filein['ns_cal_time_inds'][:] + ii*600
            plt.figure(1)
            plt.plot(time_inds, filein['ns_cal_amp'][:,ifreq,bln], 'o')
            plt.title('ns_amp for bl: %s'%filein['bl_order'][bln])
            plt.figure(2)
            y = filein['ns_cal_phase'][:,ifreq,bln]
            y[np.isfinite(y)] = np.unwrap(y[np.isfinite(y)])
            plt.plot(time_inds, y, 'o')
            plt.title('ns_phase for bl: %s'%filein['bl_order'][bln])
    except Exception as e:
        print(e)
plt.figure(1)
ylim = plt.ylim()
xlim = plt.xlim()
plt.ylim([0,ylim[1]*1.1])
plt.xlim([0,xlim[1]*1.1])

ifreq = None
for ii, filename in enumerate(file2):
    try:
        print(filename)
        with h5.File(filename, 'r') as filein:
            if ifreq is None:
                ifreq = (freq - filein.attrs['freqstart'])/filein.attrs['freqstep']
                ifreq = np.int64(np.around(ifreq))
                print(ifreq)
            plt.figure(3)
            inds = np.arange(600)
            plt.plot(inds + 600*ii, np.abs(filein['vis'][:,ifreq,bln]), 'o')
            plt.title('vis_amp for bl: %s'%filein['blorder'][bln])
            plt.figure(4)
            y = np.angle(filein['vis'][:,ifreq,bln])
            y[np.isfinite(y)] = np.unwrap(y[np.isfinite(y)])
            plt.plot(inds + 600*ii, y, 'o')
            plt.title('vis_phase for bl: %s'%filein['blorder'][bln])
    except Exception as e:
        print(e)
plt.figure(1)
plt.savefig('ns_amp')
plt.figure(2)
plt.savefig('ns_phase')
plt.figure(3)
plt.savefig('vis_amp')
plt.figure(4)
plt.savefig('vis_phase')
plt.show()






































