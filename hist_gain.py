import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import sys
'''
Histogram the difference between two data with the same shap with option for real, imag, phase, amp or total(substract and then abs)
The XY cross correlation and the correlation of the same baseline was masked.
'''

def diff_angle(a1,a2):
    da = np.abs(a1 - a2)
    da[da > np.pi] = 2*np.pi - da[da > np.pi]
    return da

#file1 = raw_input('Input the reference gain file(should be a single file):\n').strip()
#file2 = raw_input('Input the gain file for comparison(can be a single file or multiple files separated by ,):\n').strip()
if len(sys.argv) < 3:
    print('Format: python hist_gain.py reference_file comparison_file1, comparison_file2, comparison_file3...')
    sys.exit(0)
else:
    file1 = sys.argv[1].strip()
    file2 = []
    for name in sys.argv[2:]:
        file2 += [name.strip()]

gain_kind = raw_input('Input gain kind for reference file(default uni_gain):\n')
gain_kindc = raw_input('Input gain kind for comparison file(default the same as reference file):\n')
cmp_kind = raw_input('Compare total, phase, amp, imag or real?(default, total):\n')
if len(gain_kind) == 0:
    gain_kind = 'uni_gain'
if len(gain_kindc) == 0:
    gain_kindc = gain_kind
if len(cmp_kind) == 0:
    cmp_kind = 'total'
else:
    cmp_kind = cmp_kind.strip()

with h5.File(file1,'r') as filein:
    bls = filein['bl_order'][:]
for filename in file2:
    with h5.File(file1.strip(), 'r') as filein:
        gain1 = filein[gain_kind][:]
    with h5.File(filename.strip(), 'r') as filein:
        gain2 = filein[gain_kindc][:]
    for ii, (bli, blj) in enumerate(bls):
        cnan = np.nan + 1.J*np.nan
        if (bli + blj)%2 != 0:
            gain2[:,:,ii] = cnan
            gain1[:,:,ii] = cnan
        if bli == blj:
            gain2[:,:,ii] = cnan
            gain1[:,:,ii] = cnan

    if cmp_kind == 'phase':
        gain1 = np.angle(gain1)
        gain2 = np.angle(gain2)
        ratio = diff_angle(gain1,gain2)
    elif cmp_kind == 'amp':
        gain1 = np.abs(gain1)
        gain2 = np.abs(gain2)
        ratio = np.abs(gain1 - gain2)/np.abs(gain1)
    elif cmp_kind == 'real':
        ratio = np.abs(gain1.real - gain2.real)/np.abs(gain1)
    elif cmp_kind == 'imag':
        ratio = np.abs(gain1.imag - gain2.imag)/np.abs(gain1)
    else:
        ratio = np.abs(gain1 - gain2)/np.abs(gain1)
    print(filename+':')
    unmask = np.isfinite(ratio)
    mask_ratio = 1 - unmask.sum()/1./np.size(unmask)
    print('Mask ratio: %.2f'%mask_ratio)
    ratio = ratio[unmask]
    for i in range(1,6):
        max_bins = np.median(ratio)*i
        ratio_in = (ratio < max_bins).sum()/1./np.size(ratio)
        print('Ratio of data in range [0, %.2f): %.2f'%(max_bins, ratio_in))
    print('Ratio of data in plot range [0, %.2f): %.2f'%(max_bins, ratio_in))
    bins = np.linspace(0,max_bins,81)
    hist, bins = np.histogram(ratio, bins = bins, density = True)
    bins = (bins[1:] + bins[:-1])/2.
    plt.plot(bins,hist,'o',label=filename)
    plt.xlabel('Difference of '+cmp_kind)
    plt.ylabel('Density')
#    plt.hist(ratio)
plt.legend()
save_flag = raw_input('Save figure to ratio_hist.png?(y/n)').strip()
if save_flag == 'y':
    plt.savefig('ratio_hist')
plt.show()






