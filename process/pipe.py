import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import time
import subprocess as sp

filelist = []
for i in range(180):
    if i < 10:
        filelist += ['../../data/long_test/Cas_0%d.hdf5'%i] # '2018-01-01 20:44:15.844609' to '2018-01-01 21:54:15.844609'
    else:
        filelist += ['../../data/long_test/Cas_%d.hdf5'%i] # '2018-01-01 20:44:15.844609' to '2018-01-01 21:54:15.844609'

ii = 0
jj = 0
skip = []
info = 0
while(1):
    filename = filelist[ii]
    print(filename)
    files = glob('./Cas*.hdf5')
    if jj > 500:
        break
    if len(files) != 0 and not ii in skip:
        jj += 1
        time.sleep(3)
    else:
        p = sp.call('cp %s ./'%filename, shell=True)
        if p==0:
            ii += 1
            jj = 0
        else:
            time.sleep(3)

