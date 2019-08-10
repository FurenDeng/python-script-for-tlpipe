import numpy as np
import h5py as h5
import datetime
import scipy.constants as const
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

'''
input an interpolator object, return a function that can linear extrapolate.
'''
def extrap1d(interpolator, edge_dx = None):
    xs = interpolator.x
    ys = np.array(interpolator.y)
    axis = interpolator.axis
    axes = np.arange(ys.ndim).tolist()
    del axes[axis]
    axes = [axis] + axes
    ys = np.transpose(ys, axes)
    reaxes = np.arange(1,ys.ndim).tolist()
    if axis == -1:
        reaxes.append(0)
    elif axis < 0:
        reaxes.insert(axis + 1, 0)
    else:
        reaxes.insert(axis, 0)

    if edge_dx is None:
        xs1 = xs[1]
        ys1 = ys[1]
        xs_2 = xs[-2]
        ys_2 = ys[-2]
    else:
        xs1 = xs[0] + edge_dx
        ys1 = interpolator(xs1)
        xs_2 = xs[-1] - edge_dx
        ys_2 = interpolator(xs_2)

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys1-ys[0])/(xs1-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys_2)/(xs[-1]-xs_2)
        else:
            return interpolator(x)

    def ufunclike(xs):
        res = np.array(map(pointwise, np.array(xs)))
        return np.transpose(res, reaxes)

    return ufunclike

if __name__ == '__main__':
    x = np.linspace(-np.pi,np.pi,6)
#    np.random.shuffle(x)
    y = np.sin(x)
    y += np.random.rand(x.shape[0])*0.1
    intf = interp1d(x, y, kind = 'cubic')
    extf = extrap1d(intf, 1.e-3)
    xe = np.linspace(-1.2*np.pi, 1.2*np.pi, 1001)
    ye = extf(xe)
    plt.plot(x,y,'o')
    plt.plot(xe,ye)
    plt.show()
