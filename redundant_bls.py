import numpy as np
import aipy
import ephem
import calibrators as cal
import h5py as h5
import datetime
import scipy.constants as const
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from extrap1d import extrap1d
import matplotlib.pyplot as plt
from scipy.stats import linregress

datafile = 'obs_data.hdf5'
gainfile = './testdir/ns_cal/gain.hdf5'
srcname = 'cas'
src = cal.get_src(srcname)
obs = ephem.Observer()

with h5.File(datafile, 'r') as filein:
    print('Load data file!')
    obs.lon = ephem.degrees('%.10f'%filein.attrs['sitelon'])
    obs.lat = ephem.degrees('%.10f'%filein.attrs['sitelat'])
    src.compute(obs)
    sec1970 = filein.attrs['sec1970']
    inttime = filein.attrs['inttime']
    time_len = filein['vis'].shape[0]
    time_arr = np.float128(np.arange(time_len)*inttime) + sec1970
    utc_time = [datetime.datetime.utcfromtimestamp(time) for time in time_arr]
    eph_time = [ephem.date(time) for time in utc_time]
    vis = filein['vis'][:,310:510:60,:]
    pos = filein['feedpos'][:]

vis_raw = vis.copy()

n0 = []
for time in eph_time:
    obs.date = time
    src.compute(obs)
    n0.append(src.get_crds('top'))
n0 = np.array(n0) # (time, 3)

with h5.File(gainfile, 'r') as filein:
    print('Load gain file!')
    gain = filein['uni_gain'][:]
    bls = filein['bl_order'][:]
    freq = filein['freq'][:]
    time_inds = filein['ns_cal_time_inds'][:]

print('Start interpolation!')
phase = np.angle(gain)
phase_itp = interp1d(time_inds, phase, axis = 0, kind = 'cubic')
phase_exp = extrap1d(phase_itp)
newphase = phase_exp(np.arange(time_len)) # (time, freq, bl)

amp = np.abs(gain)

amp_itp = interp1d(time_inds, amp, axis = 0, kind = 'cubic')
amp_exp = extrap1d(amp_itp)
newamp = amp_exp(np.arange(time_len)) # (time, freq, bl)

exfactor = []

print('Calculate Gij!')
rij = []
for bli, blj in bls:
    fi = int(np.ceil(bli/2.))
    fj = int(np.ceil(blj/2.))
    rij += [tuple(pos[fi-1]-pos[fj-1])]
    uij = (pos[fi-1] - pos[fj-1])[:,np.newaxis] * freq[np.newaxis,:] * 1.e6 / const.c # (3, freq)
    ef = np.exp(np.dot(n0, uij)*2.J*np.pi) # (time, freq)
    exfactor += [ef]
print('Apply gain!')

axes = np.arange(1,vis.ndim).tolist()
axes.append(0)
exfactor = np.transpose(exfactor, axes)
print(exfactor.shape)
print(vis.shape)
print(newphase.shape)

with h5.File('testdir/src_vis/cas_vis.hdf5', 'r') as filein:
    src_vis = filein['src_vis'][:]
    src_rsp = np.zeros([src_vis.shape[0], src_vis.shape[1], vis.shape[-1]], dtype = np.complex64)
    for ii, (bli, blj) in enumerate(bls):
        if (bli + blj)%2 == 0:
            src_rsp[:,:,ii] = src_vis[:, :, bli%2-1, int(np.ceil(bli/2.)) - 1, int(np.ceil(blj/2.)) - 1]
        else:
            src_rsp[:,:,ii] = np.nan + 1.J*np.nan
            vis[:,:,ii] = np.nan + 1.J*np.nan
            vis_raw[:,:,ii] = np.nan + 1.J*np.nan


num_bins = 1001
exfactor = 1.
vis_raw /= exfactor
vis = vis/exfactor/np.exp(1.J*newphase)/newamp

printout = False
redun_count = 0
time_point = 700
double_count = 0
for r in set(rij):
    redun_count += 1
    if r==(0,0,0):
        continue
    arrij = np.array(rij)
    index = np.logical_and(np.logical_and(arrij[:,0] == r[0], arrij[:,1] == r[1]), arrij[:,2] == r[2])
    mask = np.isnan(vis[time_point,1,index].flatten())

    plt.figure(1)
    data = vis[time_point,1,index].flatten()
    data = data[~mask]
    if len(data) < 2:
        continue
    double_count += 1
    slope, inte, rcoe, _, _ = linregress(data.real, data.imag)
    if abs(rcoe) < 2:
        iii = np.where(index)[0]
        print('bl:\n%s'%bls[iii])
        print('r: ', r)
        print('after calibration')
        print('slope: ', slope)
        print('intercept: ', inte)
        print('corelation coefficient: ', rcoe)
        printout = True

        plt.plot(data.real, data.imag, 'o')
        plt.title('after calibration')

    plt.figure(2)
    data = vis_raw[time_point,1,index].flatten()
    data = data[~mask]

    slope, inte, rcoe, _, _ = linregress(data.real, data.imag)
    if printout:
        print('before calibration')
        print('slope: ', slope)
        print('intercept: ', inte)
        print('corelation coefficient: ', rcoe)

        plt.plot(data.real, data.imag, 'o')
        plt.title('before calibration')


    plt.figure(3)
    data = src_rsp[40,1,index].flatten()
    data = data[~mask]

    slope, inte, rcoe, _, _ = linregress(data.real, data.imag)
    if printout:
        printout = False
        print('V0')
        print('slope: ', slope)
        print('intercept: ', inte)
        print('corelation coefficient: ', rcoe)

        plt.plot(data.real, data.imag, 'o')
        plt.title('V0')

#    if double_count%6 == 0:
#        plt.show()

plt.show()
print(redun_count)













