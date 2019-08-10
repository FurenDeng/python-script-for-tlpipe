import numpy as np
import h5py as h5

def bl2pol_feed(bls, badchn = None):
    '''
        convert bls to pols and feeds
        Parameters:
        ----------
        bls: (N,2) array, baselines
        badchn: N-D array or None, bad channels, default None

        Return:
        ----------
        pf: list, [blorder, poli, feedi, plj, feedj], feeds should -1 if apply to an array
    '''
    pf = []
    if badchn is not None:
        badchn = np.int64(np.ceil(np.array(badchn)/2.))
        badchn = np.sort(list(set(badchn))) # convert bad channels to bad feeds
    else:
        badchn = np.array([])
    for bli, blj in bls:
        fi = int(np.ceil(bli/2.))
        fj = int(np.ceil(blj/2.))
        if (fi in badchn) or (fj in badchn):
            continue
        else:
            ipol = bli%2 - 1
            isearch = np.searchsorted(badchn, fi)
            i = fi - isearch# - 1
            jpol = blj%2 - 1
            jsearch = np.searchsorted(badchn, fj)
            j = fj - jsearch# - 1
            pf += [[ipol, i, jpol, j]]
    return pf
if __name__ == '__main__':
    with h5.File('./data/3srcNP_20180101214415_20180101224415.hdf5','r') as filein:
        bls = filein['blorder'][:]
        badchn = filein['channo'].attrs['badchn'][:]
        bl2pol_feed(bls, badchn)
    
