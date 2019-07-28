import matplotlib.pyplot as plt
import glob
import numpy as np
import sys

dirname = sys.argv[1]
for figname in glob.glob(dirname + '/*.png'):
    print(figname)
    im = plt.imread(figname)
    plt.imshow(im)
    plt.show(block = False)
    cmd = raw_input('q(quit)?')
    if cmd == 'q':
        plt.close()
        break
    else:
        plt.close()
