import matplotlib.pyplot as plt
import numpy as np
import h5py as h5

data = h5.File('cas_vis.hdf5','r')
src = np.abs(data['src_vis'][:])
sky = np.abs(data['sky_vis'][:])
outlier = np.abs(data['outlier_vis'][:])
freq = data.attrs['freq']
feeds = data.attrs['feed']
for k in np.arange(0,freq.shape[0],10):
    for i in np.arange(2,feeds.shape[0],5):
        for j in np.arange(2,feeds.shape[0],5):
            plt.plot(src[:,k,0,i,j], 'b-', label = 'src')
            plt.plot(sky[:,k,0,i,j], 'ro', label = 'sky')
            plt.plot(outlier[:,k,0,i,j], 'g*', label = 'outlier')
            plt.legend()
            plt.title('freq_%.2f_bl_%d_%d_xx'%(freq[k],feeds[i],feeds[j]))
            plt.show()
            plt.close()
for k in np.arange(0,freq.shape[0],10):
    for i in np.arange(0,feeds.shape[0],15):
        for j in np.arange(0,feeds.shape[0],15):
            plt.plot(src[:,k,0,i,j], 'b-', label = 'src')
            plt.plot(sky[:,k,0,i,j], 'ro', label = 'sky')
            plt.plot(outlier[:,k,0,i,j], 'g*', label = 'outlier')
            plt.legend()
            plt.title('freq_%.2f_bl_%d_%d_yy'%(freq[k],feeds[i],feeds[j]))
            plt.show()
            plt.close()
