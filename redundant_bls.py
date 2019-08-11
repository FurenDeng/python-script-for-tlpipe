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
#from bl2pol_feed import bl2pol_feed
from bl2pol_feed_inds import bl2pol_feed_inds

'''
    after calibration, vis of the redundant baseline(with the same uij) should be at the same point since the A of different feed is almost the same.

    this script plot the vis at the same time and frequency point. and the color and marker of redundant baseline is the same.

    the XY and YX correlation was not calibrated, and the XX and YY correlations of the same feed are always real, so was removed from histogram. the data points with zero amplitude should be bad points and are removed.

    seen from the result, that after calibration, points of redundant baseline get closer.
'''

calibrator = raw_input('Input calibrator name:\n') # the abbreviation of the transit source name. the whole list see the calibrators.py. used to find the ps gain file
srcname = raw_input('Input transit source:\n') # the transit point source
if len(srcname) == 0:
    srcname = calibrator
datafile = 'obs_data.hdf5' # the observed vis
ns_gainfile = './testdir/ns_cal/gain.hdf5' # the ns_cal_gain file, which also contains the united gain
ps_gainfile = 'testdir/gain/%s_gain.hdf5'%calibrator
src = cal.get_src(srcname)
obs = ephem.Observer()

with h5.File(ns_gainfile, 'r') as filein:
    print('Load gain file!')
    gain = filein['uni_gain'][:]
    blg = filein['bl_order'][:]
    freq = filein['freq'][:]
    time_inds = filein['ns_cal_time_inds'][:]

with h5.File(datafile, 'r') as filein:
    print('Load data file!')
    obs.lon = ephem.degrees('%.10f'%filein.attrs['sitelon'])
    obs.lat = ephem.degrees('%.10f'%filein.attrs['sitelat'])
    bls = filein['blorder'][:]
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
#    freq_index = np.array([freqi in freq for freqi in freq_arr])
#    vis = filein['vis'][:,freq_index,:]
    pos = filein['feedpos'][:]
    vis_shape = filein['vis'].shape
    print('frequency index range: 0 to %d'%freq.shape[0])
    freq_point = raw_input('Input frequency point(should be scalar), example(by default): %d\n'%(freq.shape[0]/2))

    print('time index range: 0 to %d'%vis_shape[0])
    time_point = raw_input('Input time point(can be scalar or multiple scalar separated by ,), example(by default): %d\n'%(vis_shape[0]/2))
    if len(freq_point) == 0:
        freq_point = freq.shape[0]/2
    else:
        freq_point = 'freq_point = ' + freq_point 
        exec(freq_point)
    if len(time_point) == 0:
        time_point = [vis_shape[0]/2]
    else:
        time_point = 'time_point = [' + time_point + ']'
        exec(time_point)
    freq_index = np.where(freq_arr == freq[freq_point])[0]
    vis = filein['vis'][:,freq_index,:]
    gain = gain[:,freq_point,:]

with h5.File(ps_gainfile, 'r') as filein:
    ps_gain = filein['gain'][:] # freq pol feed
    feeds = filein['gain'].attrs['feed']

pf, bl_select = bl2pol_feed_inds(bls, feeds)
vis = vis[...,bl_select]
vis_ps = vis.copy()
vis_raw = vis.copy()
bls = bls[bl_select]

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

vis_ps = vis_raw.copy()
rij = []
for ii,(pi,fi,pj,fj) in enumerate(pf):
    rij += [tuple(pos[fi]-pos[fj])]
    vis_ps[:,ii] /= ps_gain[None,freq_point,pi,fi] * ps_gain[None,freq_point,pj,fj].conj()
    if (pi + pj)%2 == 0 and not(pi == pj and fi == fj):
        pass
    else:
        vis_ps[:,ii] = np.nan + 1.J*np.nan
        vis[:,ii] = np.nan + 1.J*np.nan
        vis_raw[:,ii] = np.nan + 1.J*np.nan
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
    for i in np.arange(0,gain.shape[-1]-100,100):
        print('baselines:\n%s'%bls[i:i+100:20])
        plt.subplot(211)
        plt.plot(time_inds, amp[:,i:i+100:20],'o')
        plt.plot(newamp[:,i:i+100:20])
        plt.title('Amplitude')
        plt.subplot(212)
        plt.plot(time_inds, phase[:,i:i+100:20],'o')
        plt.plot(newphase[:,i:i+100:20])
        plt.title('phase')
        plt.show()

print('Calculate Gij!')
print('Apply gain!')

print(vis.shape)
print(newphase.shape)
print(vis_ps.shape)

vis = vis/np.exp(1.J*newphase)/newamp

# printout = False
redun_count = 0
double_count = 0

cmls = color_marker_line(l = False)
save_flag = raw_input('Save figure to time_point_vis/raw/ps?(y/n)').strip()
equal_flag = raw_input('Equalize x scale and y scale?(y/n)').strip()
normalize_flag = raw_input('Normalize radius?(y/n)').strip()

for tp in time_point:
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    for cml, r in zip(cmls, set(rij)):
        if tp == time_point[0]:
            redun_count += 1
        if r==(0,0,0): # XX and YY of the same feed were excluded
            continue
        arrij = np.array(rij)
        index = np.logical_and(arrij[:,0] == r[0], arrij[:,1] == r[1], arrij[:,2] == r[2])
        
        data = vis[tp,index].flatten()
        mask = np.logical_or(np.isnan(data), data == 0)
        data = data[~mask]
        if len(data) < 2:
            continue
        if tp == time_point[0]:
            double_count += 1

        plt.figure(str(tp)+'_vis')
        if normalize_flag == 'y':
            plt.plot(data.real/np.abs(data), data.imag/np.abs(data), cml)
        else:
            plt.plot(data.real, data.imag, cml)
        plt.title('after calibration')
        ylim = plt.ylim()
        xlim = plt.xlim()
        if ylim[0] < ymin:
            ymin = ylim[0]
        if ylim[1] > ymax:
            ymax = ylim[1]
        if xlim[0] < xmin:
            xmin = xlim[0]
        if xlim[1] > xmax:
            xmax = xlim[1]

        plt.figure(str(tp)+'_raw')
        data = vis_raw[tp,index].flatten()
        mask = np.logical_or(np.isnan(data), data == 0)
        data = data[~mask]

        if normalize_flag == 'y':
            plt.plot(data.real/np.abs(data), data.imag/np.abs(data), cml)
        else:
            plt.plot(data.real, data.imag, cml)
        plt.title('before calibration')
        ylim = plt.ylim()
        xlim = plt.xlim()

        plt.figure(str(tp)+'_ps')
        data = vis_ps[tp,index].flatten()
        mask = np.logical_or(np.isnan(data), data == 0)
        data = data[~mask]

        if normalize_flag == 'y':
            plt.plot(data.real/np.abs(data), data.imag/np.abs(data), cml)
        else:
            plt.plot(data.real, data.imag, cml)
        plt.title('only ps calibration')
        ylim = plt.ylim()
        xlim = plt.xlim()
        if ylim[0] < ymin:
            ymin = ylim[0]
        if ylim[1] > ymax:
            ymax = ylim[1]
        if xlim[0] < xmin:
            xmin = xlim[0]
        if xlim[1] > xmax:
            xmax = xlim[1]

    
    if equal_flag:
        plt.figure(str(tp)+'_vis')
        plt.xlim([1.1*min(xmin,ymin),1.1*max(ymax,xmax)])
        plt.ylim([1.1*min(xmin,ymin),1.1*max(ymax,xmax)])
        plt.figure(str(tp)+'_ps')
        plt.xlim([1.1*min(xmin,ymin),1.1*max(ymax,xmax)])
        plt.ylim([1.1*min(xmin,ymin),1.1*max(ymax,xmax)])
        plt.figure(str(tp)+'_raw')
    if save_flag:
        plt.savefig(str(tp)+'_ps')
        plt.savefig(str(tp)+'_raw')
        plt.savefig(str(tp)+'_vis')
    plt.show()
print('Redundant baselines count: %d'%redun_count)
print('Redundant baseline that have more than one baseline: %d'%double_count)













