import numpy as np
import h5py as h5
import sys

filename = sys.argv[1]
position = int(float(sys.argv[2]))
with h5.File(filename,'r+') as filein:
    vis = filein['vis']
    del filein['vis']
    filein.create_dataset('vis', data = np.delete(vis, position, axis = 0))

