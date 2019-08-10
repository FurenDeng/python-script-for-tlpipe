import numpy as np
import h5py as h5
from glob import glob
import matplotlib.pyplot as plt

file1 = ['save_result/3src_%d.hdf5_output'%i for i in range(84)]
file2 = ['../../data/cp_long_test/3src_%d.hdf5'%i for i in range(84)]
for ii, filename in enumerate(file1):
    try:
        with h5.File(filename + '/ns_cal/gain.hdf5', 'r') as filein:
            print(filename)
            time_inds = filein['ns_cal_time_inds'][:] + ii*600
            plt.figure(1)
            plt.plot(time_inds, filein['ns_cal_amp'][:,1,450], 'o')
            plt.title('amp')
            plt.figure(2)
            plt.plot(time_inds, filein['ns_cal_phase'][:,1,450], 'o')
            plt.title('phase')
    except Exception as e:
        print(e)

for ii, filename in enumerate(file2):
    try:
        print(filename)
        with h5.File(filename, 'r') as filein:
            plt.figure(3)
#            inds = np.where(np.abs(filein['vis'][:3600,250,200]) > 200)[0]
            inds = np.arange(600)
            plt.plot(inds + 600*ii, np.abs(filein['vis'][inds,250,450]), 'o')
            plt.title('ns_amp')
            plt.figure(4)
            plt.plot(inds + 600*ii, np.angle(filein['vis'][inds,250,450]), 'o')
            plt.title('ns_phase')
    except Exception as e:
        print(e)
#            plt.figure(5)
#            inds = np.where(np.abs(filein['vis'][:3600,250,200]) < 200)[0][::100]
#            plt.plot(inds + 3600*ii, np.abs(filein['vis'][inds,250,200]), 'o')
#            plt.title('vis_amp')
#            plt.figure(6)
#            plt.plot(inds + 3600*ii, np.angle(filein['vis'][inds,250,200]), 'o')
#            plt.title('vis_phase')
plt.show()






































