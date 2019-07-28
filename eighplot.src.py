import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import linalg as la
import h5py as h5

# visname = 'src_vis'
visname = 'sky_vis'
vis = h5.File('cas_vis.hdf5', 'r')[visname][:]
vis = np.where(np.isnan(vis),0,vis)
Vmatxx = vis[:,:,0,:,:]
Vmatyy = vis[:,:,1,:,:]

Vmatxx = Vmatxx.reshape([-1,Vmatxx.shape[-2],Vmatxx.shape[-1]])
Vmatyy = Vmatyy.reshape([-1,Vmatyy.shape[-2],Vmatyy.shape[-1]])

for V in Vmatxx[::10,:,:]:
    w,v = la.eigh(V)
    plt.figure('src')
    plt.plot(w,'.')
    plt.show()
