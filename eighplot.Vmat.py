import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import linalg as la

Vmat = np.load('Vmat.npy')
Vmat = np.where(np.isnan(Vmat), 0, Vmat)
print(Vmat.shape)
for V in Vmat[::10,:,:]:
    w,v = la.eigh(V)
    plt.figure('Vmat')
    plt.plot(w,'.')
    plt.show()
