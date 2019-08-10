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
from bl2pol_feed_inds import bl2pol_feed_inds

'''
at transit point, imag part of vis / exp(2J*pi*uij) / g should be fairly small, i.e. phase of vis / exp(2J*pi*uij) / g should be about 0

this script plot the histogram of the phase of vis / exp(2J*pi*uij) / g and vis / exp(2J*pi*uij), labeled as vis and raw vis respectively and the result shows that after calibration, more points concentrate to 0 phase.

sample of histogram are points in all the time, frequency and baseline. the XY and YX correlation was not calibrated, and the XX and YY correlations of the same feed are always real, so was removed from histogram. the data points with zero amplitude should be bad points and are removed.
'''

calibrator = raw_input('Input calibrator name:\n') # the abbreviation of the transit source name. the whole list see the calibrators.py. used to find the ps gain file
srcname = raw_input('Input transit source:\n') # the transit point source
if len(srcname) == 0:
    srcname = calibrator
datafile = 'obs_data.hdf5' # the observed vis
gainfile = './testdir/ns_cal/gain.hdf5' # the ns_cal_gain file, which also contains the united gain
ps_gain_file = 'testdir/gain/%s_gain.hdf5'%calibrator
src = cal.get_src(srcname)
obs = ephem.Observer()

#exclude_bad = False

with h5.File(gainfile, 'r') as filein:
    print('Load gain file!')
    gain = filein['uni_gain'][:]
    blg = filein['bl_order'][:]
    freq = filein['freq'][:]
    time_inds = filein['ns_cal_time_inds'][:]
# test re-arrange
#bl_range = np.arange(406)
#np.random.shuffle(bl_range)
#np.random.shuffle(bl_range)
#np.random.shuffle(bl_range)
#gain = gain[...,bl_range]
#blg = blg[bl_range]

with h5.File(datafile, 'r') as filein:
    print('Load data file!')
    bls = filein['blorder'][:]
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

with h5.File(ps_gain_file,'r') as filein:
    ps_gain = filein['gain'][:]
    feeds = filein['gain'].attrs['feed']

pf, bl_select = bl2pol_feed_inds(bls, feeds)
vis = vis[...,bl_select]
vis_ps = vis.copy()
vis_raw = vis.copy()
bls = bls[bl_select]
# re-arrange uni_gain if is not the same with data
if not np.all(bls == blg):
    print('re-arrange bls in uni_gain!')
    re_arrange_ns = []
    for bli, blj in bls:
        try:
            re_arrange_ns += [np.where(np.logical_and(blg[:,0]==bli, blg[:,1]==blj))[0][0]]
        except IndexError:
            print('blorder in data is not compatible with blorder in uni_gain!')
    re_arrange_ns = np.array(re_arrange_ns)
    gain = gain[...,re_arrange_ns]

for ii,(pi,fi,pj,fj) in enumerate(pf):
    vis_ps[:,:,ii] /= ps_gain[None,:,pi,fi]*ps_gain[None,:,pj,fj].conj()
n0 = []
for time in eph_time:
    obs.date = time
    src.compute(obs)
    n0.append(src.get_crds('top'))
n0 = np.array(n0) # (time, 3)


phase = np.angle(gain)
phase = np.unwrap(phase,axis=0)
amp = np.abs(gain)

itp_kind = raw_input('Input interpolation kind: ')
print('Start interpolation!')
if len(itp_kind) == 0:
    phase_itp = interp1d(time_inds, phase, axis = 0)
    amp_itp = interp1d(time_inds, amp, axis = 0)
else:
    phase_itp = interp1d(time_inds, phase, axis = 0, kind = itp_kind)
    amp_itp = interp1d(time_inds, amp, axis = 0, kind = itp_kind)

phase_exp = extrap1d(phase_itp, edge_dx = 1)
newphase = phase_exp(np.arange(time_len)) # (time, freq, bl)
amp_exp = extrap1d(amp_itp, edge_dx = 1)
newamp = amp_exp(np.arange(time_len)) # (time, freq, bl)

plot_itp = raw_input('Plot result of interpolation?(y/n)')
if plot_itp.strip() == 'y':
    print('baselines:\n%s'%bls[np.arange(0,gain.shape[-1],100)])
    for i in range(0, vis.shape[1], int(np.ceil(vis.shape[1]/3.))):
        print('freq index: %d'%i)
        plt.subplot(211)
        plt.plot(time_inds, amp[:,i,np.arange(0,gain.shape[-1],100)],'o')
        plt.plot(newamp[:,i,np.arange(0,gain.shape[-1],100)])
        plt.title('Amplitude')
        plt.subplot(212)
        plt.plot(time_inds, phase[:,i,np.arange(0,gain.shape[-1],100)],'o')
        plt.plot(newphase[:,i,np.arange(0,gain.shape[-1],100)])
        plt.title('phase')
        plt.show()

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
vis_ps /= exfactor
vis = vis/exfactor/np.exp(1.J*newphase)/newamp

time_range = raw_input('input time_range:\n')
if len(time_range) != 0:
    exec('time_range = [' + time_range + ']')
    time_range = np.arange(*time_range)
    vis = vis[time_range]
    vis_raw = vis_raw[time_range]
    vis_ps = vis_ps[time_range]
mask = np.logical_or(np.isnan(vis), np.abs(vis) == 0)
vis_raw = vis_raw[~mask]
vis = vis[~mask]
vis_ps = vis_ps[~mask]


hist,bins = np.histogram(np.angle(vis.flatten()), bins = np.linspace(-np.pi,np.pi,num_bins,endpoint = True))
bins = (bins[:-1] + bins[1:])/2.
plt.plot(bins, hist, 'o', label = 'vis')

hist,bins = np.histogram(np.angle(vis_raw.flatten()), bins = np.linspace(-np.pi,np.pi,num_bins,endpoint = True))
bins = (bins[:-1] + bins[1:])/2.
plt.plot(bins, hist, 'o', label = 'raw_vis')

hist,bins = np.histogram(np.angle(vis_ps.flatten()), bins = np.linspace(-np.pi,np.pi,num_bins,endpoint = True))
bins = (bins[:-1] + bins[1:])/2.
plt.plot(bins, hist, 'o', label = 'ps_gain_vis')

plt.legend()

plt.yscale('log')
plt.xlabel('phase/radian')
plt.ylabel('Count')
# plt.savefig('real_check')
plt.show()













