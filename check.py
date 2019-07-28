import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import os
import sys


'''
check the visibility of a .hdf5 file, display some information or draw the waterfall

input form: filename mode (pol) baseline (timeind = [timestart, timeend, timestep = 1])
filename:
    dish: '/datalinks/dish_1708/NGC4993_20170903' + '/NGC4993_20170903233801_20170904003757.hdf5'
    i: ./file_i.hdf5
    others
mode:
    pol: pol and baseline
    bl: baseline
    info: display information
baseline:
    pol(list or int) baseline(list or int or tuple, tuple means np.arange(*baseline)) if mode == 'pol'
    list form, like [1,2,3] or int if mode == 'bl'
timeind
    timestart: int
    timeend: int
    timestep: int

'''
pol_dict1 = {(1,1):'xx',(1,0):'xy',(0,1):'yx',(0,0):'yy'}
pol_dict2 = {0:'xx',1:'yy',2:'xy',3:'yx'}
pol = -1
timeind = 0

filename = sys.argv[1]
if filename == 'dish':
    filename = '/datalinks/dish_1708/NGC4993_20170903' + '/NGC4993_20170903233801_20170904003757.hdf5'
else:
    try:
        filename = int(float(filename))
        filename = './file_%d.hdf5'%filename
    except:
        pass

if sys.argv[2] == 'pol':
    pol = 'pol = ' + sys.argv[3]
    bl = 'bl = ' + sys.argv[4]
    exec(pol)
    exec(bl)
    if len(sys.argv) == 6:
        try:
            timeind = int(sys.argv[5])
        except:
            timeind = 'timeind = ' + 'np.arange(*' + sys.argv[5] + ')'
            exec(timeind)
elif sys.argv[2] == 'bl':
    bl = 'bl = ' + sys.argv[3]
    exec(bl)
    if len(sys.argv) == 5:
        try:
            timeind = int(sys.argv[4])
        except:
            timeind = 'timeind = ' + 'np.arange(*' + sys.argv[4] + ')'
            exec(timeind)
elif sys.argv[2] == 'info':
    with h5.File(filename,'r') as data:
        print('vis:',data['vis'])
        print('vis dimname:',data['vis'].attrs['dimname'])
        print('nfreq',data.attrs['nfreq'])
        print('freqstart',data.attrs['freqstart'])
        print('freqstep',data.attrs['freqstep'])
        print('blorder',data['blorder'])
        print('feed:',data['feedno'][:])
        try:
            print('channel: ', data['channo'][:])
            print('badchannel: ', data['channo'].attrs['badchn'])
        except:
            pass
        try:
            mask = data['vis_mask'][:]
            print('mask rate:', mask.sum()/1./len(mask.flatten()))
            #vis[mask[:]] = vis[~mask[:]].mean()
            #vis[mask[:]] = vis[~mask[:]].median()
        except Exception as aaa:
            print(aaa)
            print('mask rate:', 0)
            pass
        print('input form: filename mode (pol) baseline (timeind = [timestart, timeend, timestep = 1])')
        sys.exit(0)
if type(bl) is int:
    bl = [bl]
elif type(bl) is tuple:
    bl = np.arange(*bl)
if type(pol) is int:
    pol = [pol]

print(filename,pol,bl,timeind)

with h5.File(filename,'r') as data:
    vis = data['vis'][:]
    mask = None
    try:
        mask = data['vis_mask'][:]
        print('mask rate:', mask.sum()/1./len(mask.flatten()))
        #vis[mask[:]] = vis[~mask[:]].mean()
        vis[mask[:]] = 0.
        #vis[mask[:]] = vis[~mask[:]].median()
    except Exception as aaa:
        print('mask rate:', 0)
        pass

    if (type(timeind) is int) and timeind == 0:
        pass
    else:
        if type(timeind) is int:
            timeind = np.arange(0,vis.shape[0],timeind)
        if not (mask is None):
            mask = mask[timeind,...]
        vis = vis[timeind,...]
    print('vis:',vis.shape)
    blorder = data['blorder'][:]
    print('blorder',blorder.shape)
if pol[0] < 0:
    for blnum, i in enumerate(bl):
        figi = blnum//4
        fig = plt.figure(figi + 1)
        ax = fig.add_subplot('%d%d%d'%(1, min(4,len(bl)), blnum - min(4,len(bl))*figi))
        poli = np.array(blorder[i]).copy()
        poli = poli%2
        poli = pol_dict1[tuple(poli)]
        blo = blorder[i]/2.
        blo = np.ceil(blo)
        blo = np.array(blo, dtype = np.int32)
        ax.matshow(np.abs(vis[:,:,i]),aspect = 'equal')
        ax.set_title('baseline: (%d, %d) pol: %s'%tuple(blorder[i].tolist()+[poli]))
        if not(mask is None):
            x,y = np.where(mask[:,:,i] > 0.5)
            ax.plot(y,x,'ro')
        ax.set_xlabel('freq')
        ax.set_ylabel('time')
else:
    for i in pol:
        print(i)
        for blnum, j in enumerate(bl):
            figi = blnum//4
            fig = plt.figure('%d%d'%(i,figi + 1))
            ax = fig.add_subplot('%d%d%d'%(1, min(4,len(bl)), blnum - min(4,len(bl))*figi))
            ax.matshow(np.abs(vis[:,:,i,j]),aspect = 'equal')
            ax.set_title('baseline: (%d, %d) pol: %s'%tuple(blorder[j].tolist()+[pol_dict2[i]]))
            if not(mask is None):
                x,y = np.where(mask[:,:,i,j] > 0.5)
                ax.plot(y,x,'ro')
            ax.set_xlabel('freq')
            ax.set_ylabel('time')
plt.show(block = False)
raw_input()
plt.close('all')
