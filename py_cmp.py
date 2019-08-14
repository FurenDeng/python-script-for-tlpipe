import numpy as np
import os
import sys
from glob import glob
import subprocess as sp

'''
used to compare .py files in two dir
'''
if len(sys.argv) < 3:
    print('Format: python py_cmp.py loc_dir dist_dir')
    sys.exit(0)
else:
    loc_dir = sys.argv[1].strip()
    dist_dir = sys.argv[2].strip()

locfiles = glob(loc_dir + '/*.py')
distfiles = glob(dist_dir + '/*.py')

locbase = []
for filename in locfiles:
    locbase += [os.path.basename(filename)]

distbase = []
for filename in distfiles:
    distbase += [os.path.basename(filename)]
print('======================================')
print('local .py files:\n')
for filename in locbase:
    print(filename)
print('======================================')
print('distant .py files:\n')
for filename in distbase:
    print(filename)
newfiles = list(set(locbase) - set(distbase))
print('======================================')
print('new files:\n')
for filename in newfiles:
    print(filename)
for filename in set(locbase).intersection(distbase):
    print('======================================')
    print(filename)
    sp.call('diff %s %s'%(os.path.join(loc_dir, filename), os.path.join(dist_dir, filename)), shell = True)
