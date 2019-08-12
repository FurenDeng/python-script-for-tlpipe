import numpy as np
import h5py as h5
from bl2pol_feed_inds import bl2pol_feed_inds
import sys

'''
used to convert (freq, pol, feed) to (freq, bl)
output filename is ps filename with suffix .p2n
'''
#psfile = raw_input('Input the ps file:\n').strip()
#datafile = raw_input('Input the data file(default obs_data.hdf5, only for blorder):\n').strip()

if len(sys.argv) < 3:
    print('Format: python ps2ns.py psfile datafile\ndatafile is needed only for blorder')
    sys.exit(0)
else:
    psfile = sys.argv[1].strip()
    datafile = sys.argv[2].strip()

if len(datafile) == 0:
    datafile = 'obs_data.hdf5'
savefile = psfile + '.p2n'

with h5.File(psfile,'r') as filein:
    psgain = filein['gain'][:]
    feeds = filein['gain'].attrs['feed'][:]
with h5.File(datafile, 'r') as filein:
    bls = filein['blorder'][:]
pf, feedselect = bl2pol_feed_inds(bls, feeds)
ps2ns = np.zeros([1,psgain.shape[0],len(feedselect)], dtype = np.complex64)
bls = bls[feedselect]

for ii, (pi,fi,pj,fj) in enumerate(pf):
    ps2ns[0,:,ii] = psgain[:,pi,fi]*psgain[:,pj,fj].conj()

with h5.File(savefile, 'w') as filein:
    filein['gain'] = ps2ns
    filein['bl_order'] = bls
