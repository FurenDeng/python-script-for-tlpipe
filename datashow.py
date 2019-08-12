import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys
'''
    show observe data for specific baselines and frequencies
'''
def iterable(a):
    try:
        iter(a)
        return True
    except:
        return False
#files = glob('/datalinks/dish_1801/3srcNP_20180101/*hdf5')
if len(sys.argv) < 2:
    print('Format: python datashow.py files\nFiles will be sorted')
    sys.exit(0)
else:
    files = []
    for filename in sys.argv[1:]:
        files += [filename.strip()]
files = sorted(files)
print('Filename: \n%s'%files)
blorder = h5.File(files[0],'r')['blorder'][:]
print('freq index range %d to %d:'%(0, h5.File(files[0],'r')['vis'].shape[1]-1))
freq_inds = input('freq index: \n')
if not iterable(freq_inds):
    freq_inds = [freq_inds]
print('baseline index range %d to %d:'%(0, h5.File(files[0],'r')['vis'].shape[-1]-1))
bls = input('baselines index: \n')
if not iterable(bls):
    bls = [bls]
saveflag = raw_input('Save figure to %s?(y/n)').strip()=='y'
for bl_ind in bls:
    for freq_ind in freq_inds:
        vis = []
        print('baseline: %s'%blorder[bl_ind])
        print('freq index: %d'%freq_ind)
        bli,blj = blorder[bl_ind]
        if (bli + blj)%2 != 0:
            print('XY correlation, skip!')
            continue
        if bli == blj:
            print('Autocorrelation, skip!')
            continue
        print(bli,blj)
        for filename in files:
            vis += [np.abs(h5.File(filename, 'r')['vis'][:,freq_ind,bl_ind])]
        vis = np.concatenate(vis)
        plt.figure('bl_%s_ifreq_%s'%(blorder[bl_ind], freq_ind))
        plt.plot(vis,'o')
        if saveflag:
            plt.savefig('bl_%s_ifreq_%s'%(blorder[bl_ind], freq_ind))
        plt.show(block = False)
        raw_input('Enter to continue')
        plt.close()
