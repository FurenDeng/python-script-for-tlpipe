import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import os
'''
used to compare sky_vis, src_vis, outlier_vis separated by SPCA
'''
output_dir = 'testdir'
data_file = 'src_vis/cas_vis.hdf5'
data_file = os.path.join(output_dir, data_file)
print(data_file)

data = h5.File(data_file,'r')
freq = data.attrs['freq']
feeds = data.attrs['feed']

print('frequency index range: %d to %d'%(0, freq.shape[0]))
print('frequency range: %.4f to %.4f'%(freq[0], freq[-1]))
select_freq = raw_input('input frequency range, example (by default):  0 %d 10\n'%(freq.shape[0]))
if len(select_freq) == 0:
    select_freq = np.arange(0,freq.shape[0],10)
else:
    select_freq = select_freq.strip().replace(' ', ',')
    exec('select_freq = [' + select_freq + ']')
    select_freq = np.arange(*select_freq)


print('feed index range: %d to %d'%(0, feeds.shape[0]))
select_feedi = raw_input('input feed i range, example (by default):  0 %d 5\n'%(feeds.shape[0]))
if len(select_feedi) == 0:
    select_feedi = np.arange(1,feeds.shape[0],5)
else:
    select_feedi = select_feedi.strip().replace(' ', ',')
    exec('select_feedi = [' + select_feedi + ']')
    select_feedi = np.arange(*select_feedi)


select_feedj = raw_input('input feed j range, example (by default):  0 %d 5\n'%(feeds.shape[0]))
if len(select_feedj) == 0:
    select_feedj = np.arange(0,feeds.shape[0],5)
else:
    select_feedj = select_feedj.strip().replace(' ', ',')
    exec('select_feedj = [' + select_feedj + ']')
    select_feedj = np.arange(*select_feedj)

print('freqency: ', freq[select_freq])
print('feeds: \n', [(sfi + 1, sfj + 1) for sfi in select_feedi for sfj in select_feedj])

src = np.abs(data['src_vis'][:])
sky = np.abs(data['sky_vis'][:])
outlier = np.abs(data['outlier_vis'][:])
for k in select_freq:
    for i in select_feedi:
        for j in select_feedj:
            plt.plot(src[:,k,0,i,j], 'b-', label = 'src')
            plt.plot(sky[:,k,0,i,j], 'ro', label = 'sky')
            plt.plot(outlier[:,k,0,i,j], 'g*', label = 'outlier')
            plt.legend()
            plt.title('freq_%.2f_bl_%d_%d_xx'%(freq[k],feeds[i],feeds[j]))
            plt.show()
            plt.close()
for k in select_freq:
    for i in select_feedi:
        for j in select_feedj:
            plt.plot(src[:,k,0,i,j], 'b-', label = 'src')
            plt.plot(sky[:,k,0,i,j], 'ro', label = 'sky')
            plt.plot(outlier[:,k,0,i,j], 'g*', label = 'outlier')
            plt.legend()
            plt.title('freq_%.2f_bl_%d_%d_yy'%(freq[k],feeds[i],feeds[j]))
            plt.show()
            plt.close()
