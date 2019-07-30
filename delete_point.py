import numpy as np
import h5py as h5
import sys

'''
    delete several points at given positions from the data file to produce abnormal files
'''
filename = raw_input('input filename:\n')
position = raw_input('input the positions where one point will be deleted(should be python style command, such as [10,100,1000]):\n')
exec('position = ' + position)
with h5.File(filename,'r+') as filein:
    vis = filein['vis'][:]
    del filein['vis']
    filein.create_dataset('vis', data = np.delete(vis, position, axis = 0))

