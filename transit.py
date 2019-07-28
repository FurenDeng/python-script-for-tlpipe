import numpy as np
import sys
import ephem
import aipy
import h5py as h5
from datetime import datetime
import os
import glob
import re
class transit():
    def __init__(self, dirname, srclist = ['cyg','cas','crab']):
        self.files = sorted(glob.glob(dirname + '/*.hdf5'))
        self.srcdict = {}
        self.transit_dict = {}
        self.srclist = srclist
        with h5.File(self.files[0],'r') as data:
            #initialize the observer from data
            self.obser = ephem.Observer()
            self.obser.lon = ephem.degrees('%.10f'%data.attrs['sitelon'])
            self.obser.lat = ephem.degrees('%.10f'%data.attrs['sitelat'])
#            print(self.obser.lon)
#            print(self.obser.lat)
#            print('=========================')
            self.obser.elevation = data.attrs['siteelev']
            self.obser.epoch = data.attrs['epoch']
        for src in srclist:
            srclist, cutoff, catalogs = aipy.scripting.parse_srcs(src, 'misc')
            cat = aipy.src.get_catalog(srclist, cutoff, catalogs)
            s = cat.values()[0]
            fb = ephem.FixedBody()
            fb._ra = s._ra
            fb._dec = s._dec
            fb._epoch = s._epoch
            self.srcdict.update({src:fb.copy()})
#            print(src)
#            print(fb._ra)
#            print(fb._dec)
#            print(fb._epoch)
#            print('=========================')
    def display(self):
        for src, files in self.transit_dict.items():
            print(src+':')
            for filename in files:
                print(filename[0],filename[1])

    def run(self):
        self.transit_dict = {}
        for src in self.srclist:
            self.transit_dict.update({src:[]})
        for index, filein in enumerate(self.files):
             with h5.File(filein,'r') as data:
                self.obser.date = ephem.Date(data.attrs['obstime']) - 8*ephem.hour
                tstart = data.attrs['sec1970']# + 8*3600.
#                print('=========================================')
#                print('observer time: %s'%data.attrs['obstime'])
#                print('start time: %s'%datetime.utcfromtimestamp(tstart))

                tend = tstart + (data['vis'].shape[0] - 1)*data.attrs['inttime']
#                print('end time: %s'%datetime.utcfromtimestamp(tend))
                tend = ephem.julian_date(datetime.utcfromtimestamp(tend))
#                tephem = aipy.phs.juldate2ephem(tend)
#                print(ephem.Date(tephem))
                for src, fb in self.srcdict.items():
                    next_tran = self.obser.next_transit(fb)
                    if index == 1:
                        print(src+':', ephem.Date(next_tran))
                    next_tran = aipy.phs.ephem2juldate(next_tran)
                    if next_tran < tend:
                        self.transit_dict[src].append((index, filein))

#                print('=========================================')

tt = transit(sys.argv[1])
tt.run()
tt.display()
        

