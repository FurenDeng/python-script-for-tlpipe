import matplotlib.pyplot as plt
import glob
import numpy as np
import sys
import os

dirname = sys.argv[1]
if os.path.isdir(dirname):
    for figname in glob.glob(dirname + '/*.png'):
        print(figname)
        im = plt.imread(figname)
        plt.imshow(im)
        plt.axis('off')
        plt.show(block = False)
        cmd = raw_input('q(quit)?')
        if cmd == 'q':
            plt.close()
            break
        else:
            plt.close()
else:
    im = plt.imread(dirname)
    plt.imshow(im)
    plt.axis('off')
    plt.show()

