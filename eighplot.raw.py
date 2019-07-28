import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import linalg as la
import h5py as h5

filein = '3src_0.hdf5'
def save_xx_yy(filein):
    data = h5.File(filein,'r')
    vis = data['vis'][600:,:,:]
    vis = np.where(np.isnan(vis), 0., vis)
    bl = data['blorder'][:]
    nfeed = data.attrs['nfeeds']
    Vmatxx = np.zeros([vis.shape[0], vis.shape[1], nfeed, nfeed], dtype = np.complex64)
    Vmatyy = np.zeros([vis.shape[0], vis.shape[1], nfeed, nfeed], dtype = np.complex64)
    for index, (i, j) in enumerate(bl):
        if not(index%20):
            print(index)
        if i!=j:
            if i%2 and j%2:
                i = (i+1)/2
                j = (j+1)/2
                Vmatxx[:,:,i-1,j-1] = vis[:,:,index]
                Vmatxx[:,:,j-1,i-1] = vis[:,:,index].conj()
            if not(i%2) and not(j%2):
                i = i/2
                j = j/2
                Vmatyy[:,:,i-1,j-1] = vis[:,:,index]
                Vmatyy[:,:,j-1,i-1] = vis[:,:,index].conj()
        else:
            if i%2 and j%2:
                i = (i+1)/2
                j = (j+1)/2
                Vmatxx[:,:,i-1,j-1] = vis[:,:,index]
            if not(i%2) and not(j%2):
                i = i/2
                j = j/2
                Vmatyy[:,:,i-1,j-1] = vis[:,:,index]
    print('save file!')
    np.save('Vmatxx.npy',Vmatxx)
    np.save('Vmatyy.npy',Vmatyy)
# save_xx_yy(filein)
Vmatxx = np.load('Vmatxx.npy')
Vmatyy = np.load('Vmatyy.npy')
Vmatxx = Vmatxx.reshape([-1,Vmatxx.shape[-2],Vmatxx.shape[-1]])
Vmatyy = Vmatyy.reshape([-1,Vmatyy.shape[-2],Vmatyy.shape[-1]])
print(Vmatxx.shape)
for V in Vmatxx[::100,:,:]:
    w,v = la.eigh(V)
    plt.figure('raw')
    plt.plot(w,'.')
    plt.show()
