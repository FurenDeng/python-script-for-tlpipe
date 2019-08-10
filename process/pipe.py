import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import time
import subprocess as sp

filelist = ['../../data/cp_long_test/3src_%d.hdf5'%i for i in range(84)] # '2018-01-01 20:44:15.844609' to '2018-01-01 21:54:15.844609'

ii = 0
jj = 0
skip = []
info = 0
while(1):
    filename = filelist[ii]
    print(filename)
    files = glob('./3src*.hdf5')
    if jj > 500:
        break
    if len(files) != 0 and not ii in skip:
        jj += 1
        time.sleep(5)
    else:
        ii += 1
        jj = 0
        sp.call('cp %s ./'%filename, shell=True)

