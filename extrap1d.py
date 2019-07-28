import numpy as np
import h5py as h5
import datetime
import scipy.constants as const
from scipy.interpolate import interp1d
def extrap1d(interpolator):
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

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        res = np.array(map(pointwise, np.array(xs)))
        return np.transpose(res, reaxes)

    return ufunclike
if __name__ == '__main__':
    x = np.arange(10)
    y = [x**2]*3
    y = np.array(y)
    y = y.T # (10,3)
    y = [y]*4
    y = np.array(y, dtype = np.float64)
    y += np.random.rand(4,10,3)
    intf = interp1d(x, y, axis = 1, kind = 'cubic')
    extf = extrap1d(intf)
    x = [-2,20]
    res = extf(x)
    print(res[0,:,0])
    print(res)
    print(res.shape)
