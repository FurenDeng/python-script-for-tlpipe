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
from color_marker_line import color_marker_line

'''
after calibration, vis of the redundant baseline(with the same uij) should be at the same point since the A of different feed is almost the same.
this script plot the vis at the same time and frequency point. and the color and marker of baseline is the same.
can seen from the result, that after calibration, points of redundant baseline get closer.
'''
datafile = 'obs_data.hdf5' # the observed vis
gainfile = './testdir/ns_cal/gain.hdf5' # the ns_cal_gain file, which also contains the united gain
srcname = 'cas' # the abbreviation of the transit source name, the whole list see the calibrators.py
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

phase = np.angle(gain)
amp = np.abs(gain)

itp_kind = raw_input('Input interpolation kind: ')
print('Start interpolation!')
if len(itp_kind) == 0:
    phase_itp = interp1d(time_inds, phase, axis = 0)
    amp_itp = interp1d(time_inds, amp, axis = 0)
else:
    phase_itp = interp1d(time_inds, phase, axis = 0, kind = itp_kind)
    amp_itp = interp1d(time_inds, amp, axis = 0, kind = 'cubic')

phase_exp = extrap1d(phase_itp)
newphase = phase_exp(np.arange(time_len)) # (time, freq, bl)
amp_exp = extrap1d(amp_itp)
newamp = amp_exp(np.arange(time_len)) # (time, freq, bl)


print('Calculate Gij!')
rij = []
print('Apply gain!')

axes = np.arange(1,vis.ndim).tolist()
axes.append(0)
print(vis.shape)
print(newphase.shape)

with h5.File('testdir/src_vis/cas_vis.hdf5', 'r') as filein:
    src_vis = filein['src_vis'][:]
    src_rsp = np.zeros([src_vis.shape[0], src_vis.shape[1], vis.shape[-1]], dtype = np.complex64)
    for ii, (bli, blj) in enumerate(bls):
        fi = int(np.ceil(bli/2.))
        fj = int(np.ceil(blj/2.))
        rij += [tuple(pos[fi-1]-pos[fj-1])]
        if (bli + blj)%2 == 0:
            src_rsp[:,:,ii] = src_vis[:, :, bli%2-1, int(np.ceil(bli/2.)) - 1, int(np.ceil(blj/2.)) - 1]
        else:
            src_rsp[:,:,ii] = np.nan + 1.J*np.nan
            vis[:,:,ii] = np.nan + 1.J*np.nan
            vis_raw[:,:,ii] = np.nan + 1.J*np.nan


vis = vis/np.exp(1.J*newphase)/newamp

# printout = False
redun_count = 0
double_count = 0
print('frequency index range: 0 to %d'%vis.shape[1])
freq_point = raw_input('Input frequency point: ')

print('time index range: 0 to %d'%vis.shape[0])
time_point = raw_input('Input time point: ')
if len(freq_point) == 0:
    freq_point = 1
else:
    freq_point = int(float(freq_point))
if len(time_point) == 0:
    time_point = 700
else:
    time_point = int(float(time_point))

cmls = color_marker_line(l = False)
for cml, r in zip(cmls, set(rij)):
    redun_count += 1
    if r==(0,0,0):
        continue
    arrij = np.array(rij)
    index = np.logical_and(np.logical_and(arrij[:,0] == r[0], arrij[:,1] == r[1]), arrij[:,2] == r[2])
    mask = np.isnan(vis[time_point,freq_point,index].flatten())

    plt.figure(1)
    data = vis[time_point,freq_point,index].flatten()
    data = data[~mask]
#    if len(data) < 2:
#        continue
    double_count += 1

    plt.plot(data.real, data.imag, cml)
    plt.title('after calibration')

    plt.figure(2)
    data = vis_raw[time_point,freq_point,index].flatten()
    data = data[~mask]

    plt.plot(data.real, data.imag, cml)
    plt.title('before calibration')

    plt.figure(3)
    data = src_rsp[int(src_rsp.shape[0]/2.),1,index].flatten()
    data = data[~mask]

    plt.plot(data.real, data.imag, cml)
    plt.title('V0')

#    if double_count%6 == 0:
#        plt.show()

plt.show()
print(redun_count)













