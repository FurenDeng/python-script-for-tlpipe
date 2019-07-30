import numpy as np
import h5py as h5
import os
'''
used to check whether the uni_gain produced is correct
'''
output_dir = 'testdir'
ps_file = 'gain/cas_gain.hdf5'
ns_file = 'ns_cal/gain.hdf5'

ps_file = os.path.join(output_dir, ps_file)
ns_file = os.path.join(output_dir, ns_file)

ps_gain = h5.File(ps_file,'r')['gain'][:]
ns_phase = h5.File(ns_file,'r')['ns_cal_phase'][:]
ns_amp = h5.File(ns_file,'r')['ns_cal_amp'][:]
ns_gain = ns_amp*np.exp(1.J*ns_phase)
uni_gain = h5.File(ns_file,'r')['uni_gain'][:]
bls = h5.File(ns_file,'r')['bl_order'][:]
for ii, (bli, blj) in enumerate(bls):
    ug = uni_gain[:,:,ii]
    ng = ns_gain[:,:,ii] #t, f, bl
    pgi = ps_gain[:,bli%2-1,int(np.ceil(bli/2.))-1] #f, p, fe
    pgj = ps_gain[:,blj%2-1,int(np.ceil(blj/2.))-1]
    pg = pgi*pgj.conj()
    for uni, ns in zip(ug,ng):
        pgc = pg.copy()
        if all(np.isnan(uni)):
            continue
        if all(np.isnan(pgc)):
            continue
        if all(np.isnan(ns)):
            continue
        mask1 = np.isnan(uni)
        mask2 = np.isnan(pgc)
        mask3 = np.isnan(ns)
        mask = np.logical_or(mask2, mask3)
        uni = uni[~mask1]
        ns = ns[~mask]
        pgc = pgc[~mask]
        
        if all(pgc*ns-uni == 0):
            print('equivalent!')
        else:
            print('not equivalent!')
            print('difference: ', pg*ns - uni)

