import matplotlib.pyplot as plt
import numpy as np
import h5py as h5

kind = 'src'
data = h5.File('cas_vis.hdf5','r')
src = np.abs(data['%s_vis'%kind][:]) # 10800, 4, 2, 14, 14
# src = np.abs(data['sky_vis'][:]) # 10800, 4, 2, 14, 14
# src = np.abs(data['outlier_vis'][:]) # 10800, 4, 2, 14, 14
freq = data.attrs['freq']
feeds = data.attrs['feed']
for k in np.arange(freq.shape[0]):
    plt.figure(kind)
    for i in np.arange(0,feeds.shape[0],5):
        for j in np.arange(0,feeds.shape[0],5):
            plt.plot(src[:,k,0,i,j], label = 'bl = (%d, %d)'%(feeds[i], feeds[j]))
            plt.legend()
    plt.title('freq_%.2f_xx.png'%freq[k])
#    plt.savefig('srcfigxx/freq_%.2f_xx.png'%freq[k])
#    plt.savefig('skyfigxx/freq_%.2f_xx.png'%freq[k])
    plt.show()
    plt.close()
for k in np.arange(freq.shape[0]):
    plt.figure(kind)
    for i in np.arange(0,feeds.shape[0],5):
        for j in np.arange(0,feeds.shape[0],5):
            plt.plot(src[:,k,1,i,j], label = 'bl = (%d, %d)'%(feeds[i], feeds[j]))
            plt.legend()
    plt.title('freq_%.2f_yy.png'%freq[k])
#    plt.savefig('srcfigyy/freq_%.2f_yy.png'%freq[k])
#    plt.savefig('skyfigyy/freq_%.2f_yy.png'%freq[k])
    plt.show()
    plt.close()
