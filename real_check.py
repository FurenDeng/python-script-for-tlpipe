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

'''
imag part of vis / exp(2J*pi*uij) / g should be fairly small, i.e. phase of vis / exp(2J*pi*uij) / g should be about 0

this script plot the histogram of the phase of vis / exp(2J*pi*uij) / g and vis / exp(2J*pi*uij), labeled as vis and raw vis respectively and the result shows that after calibration, more points concentrate to 0 phase.

sample of histogram are points in all the time, frequency and baseline. the XY and YX correlation was not calibrated, and the XX and YY correlations of the same feed are always real, so was removed from histogram.
'''

datafile = 'obs_data.hdf5' # the observed vis
gainfile = './testdir/ns_cal/gain.hdf5' # the ns_cal_gain file, which also contains the united gain
srcname = 'cas' # the abbreviation of the transit source name, the whole list see the calibrators.py
src = cal.get_src(srcname)
obs = ephem.Observer()

with h5.File(gainfile, 'r') as filein:
    print('Load gain file!')
    gain = filein['uni_gain'][:]
    bls = filein['bl_order'][:]
    freq = filein['freq'][:]
    time_inds = filein['ns_cal_time_inds'][:]

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
    freqstart = filein.attrs['freqstart']
    freqstep = filein.attrs['freqstep']
    freq_arr = np.arange(freqstart, freqstart + filein['vis'].shape[1]*freqstep, freqstep, dtype = np.float32)
    freq_index = np.array([freqi in freq for freqi in freq_arr])
    vis = filein['vis'][:,freq_index,:]
    pos = filein['feedpos'][:]

vis_raw = vis.copy()

n0 = []
for time in eph_time:
    obs.date = time
    src.compute(obs)
    n0.append(src.get_crds('top'))
n0 = np.array(n0) # (time, 3)


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


# for i in range(4):
#     plt.plot(time_inds, amp[:,i,np.arange(0,512,100)],'o')
#     plt.plot(newamp[:,i,np.arange(0,512,100)])
#     plt.show()

exfactor = []

print('Calculate Gij!')
rij = []
for ii, (bli, blj) in enumerate(bls):
    fi = int(np.ceil(bli/2.))
    fj = int(np.ceil(blj/2.))
    rij += [tuple(pos[fi-1]-pos[fj-1])]
    uij = (pos[fi-1] - pos[fj-1])[:,np.newaxis] * freq[np.newaxis,:] * 1.e6 / const.c # (3, freq)
    ef = np.exp(np.dot(n0, uij)*2.J*np.pi) # (time, freq)
    exfactor += [ef]
    
    cnan = np.nan + 1.J*np.nan
    if (bli + blj)%2 != 0 or bli == blj:
        vis[:,:,ii] = cnan

print('Apply gain!')

axes = np.arange(1,vis.ndim).tolist()
axes.append(0)
exfactor = np.transpose(exfactor, axes)
print(exfactor.shape)
print(vis.shape)
print(newphase.shape)


num_bins = 1001
vis_raw /= exfactor
vis = vis/exfactor/np.exp(1.J*newphase)/newamp

vis_raw = vis_raw[~np.isnan(vis)]
vis = vis[~np.isnan(vis)]


hist,bins = np.histogram(np.angle(vis.flatten()), bins = np.linspace(-np.pi,np.pi,num_bins,endpoint = True))
bins = (bins[:-1] + bins[1:])/2.
plt.plot(bins, hist, 'o', label = 'vis')

hist,bins = np.histogram(np.angle(vis_raw.flatten()), bins = np.linspace(-np.pi,np.pi,num_bins,endpoint = True))
bins = (bins[:-1] + bins[1:])/2.
plt.plot(bins, hist, 'o', label = 'raw_vis')
plt.legend()

plt.yscale('log')
plt.xlabel('phase/radian')
plt.ylabel('Count')
plt.show()













