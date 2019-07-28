import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

# filein = '3srcNP_20180101214415_20180101224415.hdf5'
filein = '3src_1.hdf5'
def save_raw_xx_yy(filein):
    data = h5.File(filein,'r')
    vis = np.abs(data['vis'][:]) # Time Frequency Baselin
    nfeeds = data.attrs['nfeeds']
    freqstart = data.attrs['freqstart']
    freqstep = data.attrs['freqstep']
    nfreq = data.attrs['nfreq']
    freq = freqstart + np.arange(nfreq)*freqstep
    visxx = np.zeros([vis.shape[0],vis.shape[1],nfeeds*(nfeeds + 1)/2])
    visyy = np.zeros([vis.shape[0],vis.shape[1],nfeeds*(nfeeds + 1)/2])
    viscross = np.zeros([vis.shape[0],vis.shape[1],nfeeds*nfeeds])
    bl = data['blorder']
    blxx = []
    blyy = []
    blcross = []
    ixx = 0
    iyy = 0
    icross = 0
    for index,(i,j) in enumerate(bl):
        if i%2 and j%2:
            visxx[:,:,ixx] = vis[:,:,index]
            ixx += 1
            blxx += [[i,j]]
        elif (not i%2) and (not j%2):
            visyy[:,:,iyy] = vis[:,:,index]
            iyy += 1
            blyy += [[i,j]]
        else:
            viscross[:,:,icross] = vis[:,:,index]
            icross += 1
            blcross += [[i,j]]
    print('start saving result!')
    np.save('visxx',visxx)
    np.save('visyy',visyy)
    np.save('viscross',viscross)
    np.save('blxx',blxx)
    np.save('blyy',blyy)
    np.save('blcross',blcross)
    np.save('freq',freq)

# save_raw_xx_yy(filein)
kind = 'xx'
visname = 'vis%s.npy'%kind
blname = 'bl%s.npy'%kind


vis = np.load(visname)
bl = np.load(blname)
freq = np.load('freq.npy')
print(bl.shape)
for i in np.arange(0,vis.shape[1],20):
    for j in np.arange(0,vis.shape[2],20):
        plt.plot(vis[:,i,j], label = 'bl = (%d, %d)'%(bl[j][0],bl[j][1]))
        plt.legend()
    plt.ylim([-0.5,45])
    plt.title('freq_%.2f_%s.png'%(freq[i], kind))
    plt.savefig('freq_%.2f_%s.png'%(freq[i], kind))
    plt.show()
#    plt.close()
'''
visyy = np.load('visyy.npy')
blyy = np.load('blyy.npy')
freq = np.load('freq.npy')
print(blyy.shape)
for i in np.arange(0,visyy.shape[1],20):
    for j in np.arange(0,visyy.shape[2],20):
        plt.plot(visyy[:,i,j], label = 'bl = (%d, %d)'%(blyy[j][0],blyy[j][1]))
        plt.legend()
    plt.ylim([-0.5,25])
    plt.title('freq_%.2f_yy.png'%freq[i])
    plt.savefig('figyy/freq_%.2f_yy.png'%freq[i])
    plt.close()
'''




