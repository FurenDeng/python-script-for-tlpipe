import numpy as np
import h5py as h5

def ps2ns(psfile, nsfile):
    ps_data = h5.File(psfile,'r')
    ps_gain = ps_data['gain'][:] # freq, pol, feed
    ns_data = h5.File(nsfile,'r')
    ns_gain = np.exp(1.J * ns_data['ns_cal_phase'][:])# * ns_data['ns_cal_amp'][:] #time, freq, bl
    blorder = ns_data['bl_order']
    badchn = ns_data['channo'].attrs['badchn']
    for bc in badchn.copy():
        if bc%2:
            badchn = np.append(badchn, bc + 1)
        else:
            badchn = np.append(badchn, bc - 1)
    badchn = np.array(list(set(badchn)))

    pg2d = np.zeros([ps_gain.shape[0],ps_gain.shape[1]*ps_gain.shape[2]],dtype = np.complex64) # freq, feed(odd X, even Y)
    pg2d[:,::2] = ps_gain[:,0,:]
    pg2d[:,1::2] = ps_gain[:,1,:]

    pg2d = pg2d.T # feed(odd X, even Y), freq
    ng3d = pg2d[np.newaxis,:] * pg2d[:,np.newaxis].conj()
    ng3d = np.transpose(ng3d,[2,1,0])
    ng2d = np.zeros([ns_gain.shape[1],ns_gain.shape[2]], dtype = np.complex64)
    print(ng2d.shape)
    print(ng3d.shape)
    cnan = complex(np.nan, np.nan)
    for blind, (i,j) in enumerate(blorder):
        if (i in badchn) or (j in badchn):
            continue
        else:
            isearch = np.searchsorted(badchn, i)
            i = i - isearch - 1
            jsearch = np.searchsorted(badchn, j)
            j = j - jsearch - 1
            print(i,j,blind)
            ng2d[:,blind] = ng3d[:,i,j]
    new_ns_gain = ns_gain.conj()*ng2d
    return new_ns_gain
import sys
psfile = sys.argv[1]
nsfile = sys.argv[2]
res = ps2ns(psfile,nsfile)
print(res)
