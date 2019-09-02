import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.optimize import curve_fit
from fftest import fft
from scipy.signal import find_peaks_cwt
from scipy.interpolate import interp1d
from scipy.integrate import romb
from scipy.integrate import quad
import scipy as sc
import warnings
from autograd import grad
import autograd.numpy as autonp
from autograd import elementwise_grad as egrad
from autograd import primitive
from scipy.signal import find_peaks, find_peaks_cwt

def axes_trans(array, axis):
    '''convert axis to the first axis, return the axes order used in np.transpose and reverse axes order
    Parameters:
    ----------
    array: array of data that will be transposed
    axis: the axis that will be moved to the first

    Returns
    ----------
    tran_ax: used in np.transpose to transpose the array
    re_ax: used in np.transpose to convert the array back
    '''
    axes_size = np.ndim(array)
    axes = list(range(axes_size))
    index = axis%axes_size # apply to both negative and positive axis
    del axes[index]
    tran_ax = [index] + axes
    re_ax = list(range(1,axes_size))
    re_ax.insert(index, 0)
    return np.array(tran_ax), np.array(re_ax)

def sinc_fit(x, y, L, x0 = None, p_real = [1], p_imag = [1], x_ind = None, debug = False, max_iter = 5):
    '''
        Use sin((x-x0)*L)/(x-x0)/(2*pi)**0.5 + sin((x+x0)*L)/(x+x0)/(2*np.pi)**0.5 and its derivatives as a set of base to fit real and imag part of data.
        The real part and imag part are treated separately, and two set of params will be returned. However, iteration for x0 and L is used to make which of real and imag part closer.
        Notice the result is VERY sensitive to x0 and L. L can be larger a little, but should not be smaller than the actual value.
        
        Parameters:
        -----------
        x: (N,), array of x data used to fit.
        y: (N,), array of y data used to fit.
        L: scalar, initial guess for L. should be close to the actual value, can be a little bit larger.
        x0: scalar or None, initial guess for x0. if None, x0 will be set as the mean of max and min value of real and imag points. default None.
        p_real: (M,) initial guess for real part, and the length should be the order of fitting. default [1]
        p_imag: (M,) initial guess for imag part, and the length should be the order of fitting. default [1]
        x_ind: (2,), (n,) or None. if 2, only use x[x_ind[0]:x_ind[1]]. if (n,), only use x[x_ind]. if None, use the full set of data. default None.
        debug: whether debug the output. default False.
        max_iter: iterate how many times to make x0 and L for real part and imag part closer.

        Return:
        -----------
        p1: fitting result for real part, [[x0, L, *p],err]
        p2: fitting result for imag part, [[x0, L, *p],err]
    '''
#    def sinc_real(x, x0, L, *p):
#        return SincPoly(x0, L, len(p)-1, p)(x).real
#    def sinc_imag(x, x0, L, *p):
#        return SincPoly(x0, L, len(p)-1, p)(x).imag
    def sinc(x, x0, L, *p):
        return SincPoly(x0, L, len(p)-1, p, imag = False)(x)
    if x_ind is not None:
        if len(x_ind) == 2:
            x = np.array(x)[x_ind[0]:x_ind[1]]
            y = np.array(y)[x_ind[0]:x_ind[1]]
        else:
            x = np.array(x)[x_ind]
            y = np.array(y)[x_ind]
    else:
        x = np.array(x)
        y = np.array(y)
    if x0 is None:
        peak1 = x[y.real == y.real.max()]
        peak2 = x[y.real == y.real.min()]
        peak3 = x[y.imag == y.imag.max()]
        peak4 = x[y.imag == y.imag.min()]
        peaks = [peak1, peak2, peak3, peak4]
        x0 = []
        if debug:
            print('real max: %s'%peak1)
            print('real min: %s'%peak2)
            print('imag max: %s'%peak3)
            print('imag min: %s'%peak4)
        for peak in peaks:
            if len(peak) < 3:
                x0 += [np.abs(peak).mean()]
        x0 = np.mean(x0)
        # if x0 is exact a maximum or minimum, there is likely be error in fitting(essentially infinity in autogradient). round it to 1% precision.
        round_digit = -np.floor(np.log10(x0)) + 2
        x0 = np.around(x0, np.int64(round_digit))
        print('Automatically set x0 = %s'%x0)
    param_real = np.concatenate([[x0, L], p_real])
    param_imag = np.concatenate([[x0, L], p_imag])
#    p1 = curve_fit(sinc_real, x, y.real, param)
#    p2 = curve_fit(sinc_imag, x, y.imag, p1[0])
    p1 = curve_fit(sinc, x, y.real, param_real)
    p2 = curve_fit(sinc, x, y.imag, param_imag)
    if debug:
        print('result for fitting without iteration:')
        print('=====================================')
        print('result for real part:\nx0 = %s, L = %s, p = %s'%(p1[0][0],p1[0][1],p1[0][2:]))
        print('err: %s'%p1[1])
        print('result for imag part:\nx0 = %s, L = %s, p = %s'%(p2[0][0],p2[0][1],p2[0][2:]))
        print('err: %s'%p2[1])
    # do this because real and imag part cannot fit all params separately
    for i in np.arange(max_iter):
        p1 = curve_fit(sinc, x, y.real, np.concatenate([p2[0][:2],p1[0][2:]]))
        p2 = curve_fit(sinc, x, y.imag, np.concatenate([p1[0][:2],p2[0][2:]]))
    if debug:
        print('final result:')
        print('=====================================')
        print('result for real part:\nx0 = %s, L = %s, p = %s'%(p1[0][0],p1[0][1],p1[0][2:]))
        print('err: %s'%p1[1])
        print('result for imag part:\nx0 = %s, L = %s, p = %s'%(p2[0][0],p2[0][1],p2[0][2:]))
        print('err: %s'%p2[1])
        plt.figure('sinc_fit_debug')
        plt.plot(x,y.real,'r.',label='data real')
        plt.plot(x,sinc(x,*p1[0]),'r-',label='fit real')
        plt.plot(x,y.imag,'b.',label='data imag')
        plt.plot(x,sinc(x,*p2[0]),'b-',label='fit imag')
        plt.legend()
        plt.show(block = False)
        raw_input('Type any key to continue')
        plt.close('all')
    return p1, p2

def gaussian_fit(x, y, loc = 0., sigma = 1., p_poly = [1], x_ind = None, debug = False):
    '''
        Use gaussian function multiply polynomial:
            np.poly(p,x)*np.exp(-(x - loc)**2/2./sigma**2)
        to fit location and sigma of peak

        Parameters:
        -----------
        x: (N,), the x data to fit
        y: (N,), the y data to fit
        loc: scalar, initial guess for loc
        sigma: scalar, initial guess for sigma
        p_poly: (M,), initial guess for polynomial parameter p
        x_ind: (2,) or None. if 2, only use x[x_ind[0]:x_ind[1]]. if None, use the full set of data. default None.
        debug: whether debug the output. default False.

        Return:
        -----------
        p: [[loc, sigma, *p],err]
    '''
    def gaus(x, loc, sigma, *p):
        t1 = np.exp(-(x - loc)**2/2./sigma**2)
        return np.polyval(p, x)*t1
    if x_ind is not None:
        x = np.array(x)[x_ind[0]:x_ind[1]]
        y = np.array(y)[x_ind[0]:x_ind[1]]
    else:
        x = np.array(x)
        y = np.array(y)
    param = np.concatenate([[loc, sigma], p_poly])
    p = curve_fit(gaus, x, y, param)
    if debug:
        print('p:%s'%p[0])
        print('err:%s'%p[1])
        plt.figure('gaussian_fit_debug')
        plt.plot(x,y,'.',label='data')
        plt.plot(x,gaus(x,*p[0]),'-',label='fit')
        plt.legend()
        plt.show(block = False)
        raw_input('Type any key to continue')
        plt.close('all')
    return p
    
def factorial(n, kind = None):
    '''
        Improved factorial, can deal with float and int. and can deal with n > 170 which will be inf for scipy.misc.factorial. alway added double factorial feature.
        Except for kind == 'gamma', n should always larger or equal 0. and if kind != 'gamma', and n is float, n will be convert to np.int64.
        Notice the return is always float, should be convert to int if need.

        Parameters:
        -----------
        n: the input order.
        kind: 'gamma': return scipy.special.gamma(x), 'odd': n should be odd, return n!!, 'even': n should be even, return n!!, else return n!

        Return:
        -----------
        result: result of factorial
    '''
    if kind == 'gamma':
        return sc.special.gamma(n)
    if np.asarray(n).dtype.kind != 'i':
        n = np.int64(n)
    assert n >= 0, 'n should not be negative except kind = gamma!'
    if n == 0:
        return 1.
    if kind == 'odd':
        assert n%2==1, 'n should be odd instead of input %d'%n
        if n <= 299:
            return np.prod(np.arange(1,n+1,2,dtype=np.float64))
        else:
            return np.prod(np.arange(1,n+1,2,dtype=np.float128))
    if kind == 'even':
        assert n%2==0, 'n should be even instead of input %d'%n
        if n <= 300:
            return np.prod(np.arange(2,n+1,2,dtype=np.float64))
        else:
            return np.prod(np.arange(2,n+1,2,dtype=np.float128))
    if n <= 170:
        return np.prod(np.arange(1,n+1,dtype=np.float64))
    else:
        return np.prod(np.arange(1,n+1,dtype=np.float128))

def normalize_factor(m,sigma):
    '''
        Normalization factor for gaussian function multiply polynomial:
            (x - loc)**m*exp(-(x - loc)**2/2./sigma**2)
        Notice the center of polynomial should be the same as the gaussian
        
        Parameters:
        -----------
        m: scalar
        sigma: scalar

        Return:
        -----------
        normalization factor: scalar
    '''
    if m == 0:
        return sigma*(np.pi*2.)**0.5
    if m%2 != 0:
        return 0.
    else:
        return factorial(m-1,'odd')*sigma**(m+1)*(2*np.pi)**0.5

def gaussian(x, loc, sigma, p = [1], normalize = False):
    '''
        Gaussian function multiply polynomial:
            polyval(p, x - loc)*exp(-(x - loc)**2/2./sigma**2)
        the normalization factor is estimated by [-10*sigma, 10*sigma] integral.
        
        Parameters:
        -----------
        x: (N,), input x array.
        loc: scalar, location of gaussian function.
        sigma: scalar, std of gaussian function.
        p: (M,), coeffecients for polynomial, determine the order of polynomial.
        normalize: bool, if True, \int_{-\infty}^\infty gaussian will be normalize to 1.

        Return:
        ----------
        polyval(p, x - loc)*exp(-(x - loc)**2/2./sigma**2)/normalization_factor
    '''
    if normalize:
        def fun(x):
            t1 = np.exp(-(x - loc)**2/2./sigma**2)
            return np.polyval(p, x)*t1
        norm_factor = quad(fun,loc-10.*sigma**2,loc+10.*sigma**2)[0]
#        for i,p0 in enumerate(p[::-1]):
#            norm_factor += normalize_factor(i,sigma)*p0
        return fun(x)/norm_factor
    else:
        t1 = np.exp(-(x - loc)**2/2./sigma**2)
        return np.polyval(p, x)*t1

def fft1d(x, y, interp_kind = 'linear', dx = None, dk = None, axis = -1, debug = False):
    '''
        F(f(x)) = int f(x)*exp(-1.J*k*x)dx/(2*np.pi)**0.5.
        Allow for uneven distributed data. If np.diff(x).std() > np.diff(x).mean()*1.e-5, the data will be treated as uneven.
        Use interpolation when data is uneven distributed or np.abs((x[1]-x[0])/dx-1)>0.1
        Use 1/dx/dk as the n in np.fft.fft.
        Notice, the interpolation might lead to unexpected imag or real part, derivation from actual value, or dephasing for periodic function, so never set dx if dont need.

        Parameters:
        -----------
        x: (N,), x data array. np.argsort and unique will be used to it.
        y: (N,), y data array. it will be sorted together will x.
        interp_kind: string, 'kind' param in scipy.interpolate.interp1d. default linear.
        dx: scalar or None. if None, for uneven case, dx will be (x[-1]-x[0])/(x.shape[0]-1). if not None, x will be interpolate no matter what. default None.
        dk: scalar or None. the dk for output result, will change the np.fft.fft param n = 1/dx/dk, if None, dk = 1/dx/x.shape. default None.
        axis: int, the axis to do the fft. default -1.
        debug: bool, whether debug

        Return: list, [k,yk,interpf]. if no interpolation was performed, interpf = None
        ---------

    '''
    xarg = np.argsort(x)
    x = np.array(x)[xarg]
    y = np.take(y,xarg,axis)
    x, induni = np.unique(x,return_index=True)
    y = np.take(y,induni,axis)
    if debug:
        plt.figure('fft1d_debug')
        ax_T, _ = axes_trans(y, axis)
        yplot = np.transpose(y, ax_T)
        yplot = yplot.reshape(yplot.shape[0],-1)
        plt.plot(x,yplot,'.',label='data')
    if np.diff(x).std() > np.diff(x).mean()*1.e-5:
    # unevenly distribute, interpolate and then fft
        print('Detected uneven data, interpolate it!')
        if dx is None:
            dx = (x[-1] - x[0])/(x.shape[0]-1)
#            dx = np.median(np.diff(x))
#            dx = np.diff(x).mean()
#            dx = np.diff(x).min()
#            dx = np.diff(x).max()
        interpf = interp1d(x,y,kind=interp_kind,axis=axis)
        x = np.arange(x[0],x[-1]+dx,dx)
        x = x[x<=interpf.x[-1]]
        print('x range: %s'%[x[0],x[-1],dx])
        y = interpf(x)
        if debug:
            yplot = np.transpose(y, ax_T)
            yplot = yplot.reshape(yplot.shape[0],-1)
            plt.plot(x,yplot,label='interpolate')
    elif (dx is not None):
        if np.abs((x[1] - x[0])/dx - 1.) > 0.1:
        # large difference between dx and the interval in data
            print('Change the dx from %s to %s'%(x[1] - x[0],dx))
            interpf = interp1d(x,y,kind=interp_kind,axis=axis)
            x = np.arange(x[0],x[-1]+dx,dx)
            x = x[x<=interpf.x[-1]]
            print('x range: %s'%[x[0],x[-1],dx])
            y = interpf(x)
            if debug:
                yplot = np.transpose(y, ax_T)
                yplot = yplot.reshape(yplot.shape[0],-1)
                plt.plot(x,yplot,label='interpolate')
        else:
            print('Do not need to change the dx from %s to %s'%(x[1] - x[0],dx))
            print('x range: %s'%[x[0],x[-1],dx])
            dx = x[1] - x[0]
            interpf = None
    else:
        dx = x[1] - x[0]
        print('x range: %s'%[x[0],x[-1],dx])
        interpf = None
    if dk is not None:
    # in fft, dk = 1/n_x/dx
        n_c = 1./(x[1] - x[0])/dk*2.*np.pi
        n_c = np.int64(np.ceil(n_c))
        if n_c < x.shape[0]:
            warnings.warn('Input dk is smaller than the dk from the data!')
        if n_c == x.shape[0]:
            n_c = None
        else:
            print('Change the dk from %s to %s'%(1./x.shape[0]/(x[1] - x[0]),dk))
    else:
        n_c = None
    if n_c is None:
        k = np.fft.fftfreq(x.shape[0],dx)*2.*np.pi
    else:
        print('set n as %d'%n_c)
        k = np.fft.fftfreq(n_c,dx)*2.*np.pi
    if debug:
        plt.legend()
        plt.show()
    print('start fft!')
    yk = np.fft.fft(y, n_c, axis = axis)
    print('shift result!')
    rshp = np.ones(yk.ndim, int)
    rshp[axis] = -1
    shift_factor = dx*np.exp(-1.j*x[0]*k)/(np.pi*2.)**0.5
    yk = yk*shift_factor.reshape(rshp)
    return [k,yk,interpf]

def autogradientn(func, n = 1):
    '''
        Improved version for autograd.egrad, can receive int input. and calculate high order derivative.
        
        Parameters:
        -----------
        func: callable, function to calculate derivative, should be autograd function.
        n: the order of derivative.

        Return:
        -----------
        result: callable, n order derivative function.
    '''
    df = egrad(func)
    for i in np.arange(n-1):
        df = egrad(df)
    return lambda x: df(np.asarray(x)*1.)
def autogradientarr(func, n):
    '''
        Calculate 0 to n order derivative function.
        Notice never construct array of functions with the result being parts of the returned functions, which will lead to unknown error. But single functions can still be constructed.

        Parameters:
        -----------
        func: callable, function to calculate derivative, should be autograd function.
        n: the order of derivative.

        Return:
        -----------
        df: (n,) ndarray, 0 to n order derivative function. 0 means input func.
    '''
    df = np.empty(n+1,dtype='object')
    df[0] = func
    for i in np.arange(1, n + 1):
        df[i] = autogradientn(func,i)
    return df
def SincPoly(x0, L, n = 0, p = None, imag = True, return_all = False):
    '''
        Linear combination of sin((x-x0)*L)/(x-x0)/(2*pi)**0.5 + sin((x+x0)*L)/(x+x0)/(2*np.pi)**0.5 and its derivatives with p the coefficients.
        Notice L is for fft from cos or sin from finit section. L is HALF of length of the section.
        Notice n >= 0, len(p) = n + 1, else n will be change to suit len(p)

        Parameters:
        -----------
        x0: scalar, x0 parameter.
        L: scalar, L parameter.
        n: scalar, order of derivative used to build the sery. default 0.
        p: (n,) or None, coefficients for combination of different terms. if None, p=np.ones(n + 1).
        imag: bool, if True, i th derivation will times an additional factor (1.J)**i
        return_all: bool, if True, return [retfun, y] else retfun
        
        Return:
        ------------
        retfun: callable, the combination of 0 to n order derivative of sin((x-x0)*L)/(x-x0)/(2*pi)**0.5 + sin((x+x0)*L)/(x+x0)/(2*np.pi)**0.5
        y: (n,) ndarray, 0 to n order derivative of sin((x-x0)*L)/(x-x0)/(2*pi)**0.5 + sin((x+x0)*L)/(x+x0)/(2*np.pi)**0.5

    '''
    if p is None:
        p = np.ones(n + 1)
    elif n+1 != len(p):
        print('n %s is not compatible with length of p %s. reset n!'%(n, len(p)))
        n = len(p) - 1
    p = np.array(p)[::-1]
    def sinc(x):
        return autonp.sinc((x-x0)/np.pi*L)*L/(2*np.pi)**0.5 + autonp.sinc((x+x0)/np.pi*L)*L/(2*np.pi)**0.5
    y = autogradientarr(sinc, n)
    def retfun(x):
        ret = 0
        for i in np.arange(n+1):
            if imag:
                ret += y[i](x)*(1.J)**i*p[i]
            else:
                ret += y[i](x)*p[i]
        return ret
    if return_all:
        return retfun, y
    else:
        return retfun

def common_period(x,tol=0.):
    '''
        sort the element in an ndarray according to the number of multiple in the array in descending order. 0 are excluded.

        Parameters:
        -----------
        x: array with any shape. the input array.
        tol: scalar, all number between n*x-tol and n*x+tol will be considered as multiple as x, with n integer. default 0.

        Return:
        ----------
        unis: the unique element in input array x.
        pcount: the number of multiple for elements in unis, in descending order.
    '''
    x = np.array(x)
    unis, counts = np.unique(x,return_counts = True)
    csort = np.argsort(counts)
    unis = unis[csort]
    unis = unis[unis!=0]
    pcount = []
    for uni in unis:
        pcount += [np.logical_or(x%uni<=tol, (x+tol)%uni<=tol).sum()]
    psort = np.argsort(pcount)
    pcount = np.asarray(pcount)
    return unis[psort][::-1], pcount[psort][::-1]

def correlation_shift(y, window, x = None, xwin = None, mode = 'valid', debug = False, prominence_std = 0., **kwargs):
    '''
        Calculate the period of array y and the dephasing between y and window by np.correlate(y,window).
        Notice the shorter one of y and window should contain complete period to make the result precise. otherwise, the the result will shift several inds, which depends only on the uncomplete ratio. denser x doesnt help.
        
        Parameters:
        -----------
        y: (N,) array, input array to find period and dephasing.
        window: (M,) the window array.
        x: (n,) or None. array, to get dx and x[0] for y[0]. dont need to have the same shape as y. but len(x) should >= 2. if x is None or len(x) < 2, x will be set as [0, 1]. default None.
        xwin: (m,) or None. array, to get xwin[0] for window[0]. dont need to have the same shape as window, but dxwin should equal dx. and len(xwin) should >= 2. if xwin is None or len(xwin) < 2, xwin will be set as x. default None.
        mode: string, mode in np.correlate. default 'valid'.
        debug: bool, if True, debug.
        prominence_std: scalar, prominence param in scipy.signal.find_peaks will be set as prominence_std*corr.std(), to trim peaks and valleys. only work when no kwargs are given.
        **kwargs: kwargs in scipy.signal.find_peaks

        Return:
        --------
        shift_x: if set x = x-shift_x, dephasing between y and window will be compensated.
        period: period of data, should be one of pmax and pmin
        pmax: period gotten from maximum
        pmin: period gotten from minimum
    '''
    # pre-process check even, same dx, ascending
    if x is None:
        x = np.arange(2)
    elif len(x) < 2:
        warnings.warn('len(x) is shorter than 2! Change it to [0, 1]')
        x = np.arange(2)
    if xwin is None:
        xwin = np.array(x)
    elif len(xwin) < 2:
        warnings.warn('len(xwin) is shorter than 2! Change it to [0, 1]')
        xwin = np.array(x)
    if any(np.diff(x)<0):
        assert len(x) == len(y), 'The x is not ascending. length of x %s and y %s are not the same, so cannot sort them!'%(len(x),len(y))
        print('unascending x, sort it!')
        sortargs = np.argsort(x)
        x = x[sortargs]
        y = y[sortargs]
    if any(np.diff(xwin)<0):
        assert len(xwin) == len(window), 'The xwin is not ascending. length of xwin %s and window %s are not the same, so cannot sort them!'%(len(xwin),len(window))
        print('unascending xwin, sort it!')
        sortargs = np.argsort(xwin)
        print(xwin[1] - xwin[0])
        xwin = xwin[sortargs]
        print(xwin[1] - xwin[0])
        window = window[sortargs]
    dx = x[1] - x[0]
    print('dx: %s'%dx)
    assert np.abs(xwin[1]-xwin[0]-dx)<1.e-15, 'Interval of x %s and xwin %s is not the same!'%(dx, xwin[1]-xwin[0])
    print('mode: %s'%mode.strip())

    # for 'valid' and 'same' mode, y that shorter than window always lead to poor result, exchange y and window to improve result.
    if len(window) > len(y):
        if debug:
            print('len(window) > len(y), exchange them!')
        interwin = np.array(window)
        window = np.array(y)
        y = interwin
        interwinx = np.array(xwin)
        xwin = np.array(x)
        x = interwinx
        exchange = True
    else:
        window = np.array(window)
        y = np.array(y)
        x = np.array(x)
        xwin = np.array(xwin)
        exchange = False
    if mode.strip() == 'valid':
        print('displacement of x and xwin at first overlap in index: %d'%np.around((x[0] - xwin[0])/dx))
    elif mode.strip() == 'full':
        print('displacement of x and xwin at first overlap in index: %d'%np.around((x[0] - xwin[0])/dx - len(window) + 1))
    elif mode.strip() == 'same':
        print('displacement of x and xwin at first overlap in index: %d'%np.around((x[0] - xwin[0])/dx - np.ceil((len(window) - 1)/2.)))
    corr = np.correlate(y, window, mode = mode)
    if len(kwargs) == 0:
        max_inds,prop1 = find_peaks(corr, prominence = prominence_std*corr.std())
        min_inds,prop2 = find_peaks(-corr, prominence = prominence_std*corr.std())
        print('maximum prominences/std: %s'%(prop1['prominences']/corr.std()))
        print('minimum prominences/std: %s'%(prop2['prominences']/corr.std()))
    else:
        max_inds,prop1 = find_peaks(corr, **kwargs)
        min_inds,prop2 = find_peaks(-corr, **kwargs)
    if debug:
        plt.figure('data and window')
        plt.plot(y,'r.',label='data')
        plt.plot(window,'b-',label='window')
        plt.legend()
        plt.figure('correlation_debug')
        plt.plot(corr, label = 'correlation')
        plt.plot(max_inds, corr[max_inds],'o',label='maximum')
        plt.plot(min_inds, corr[min_inds],'o',label='minimum')
        if 'prominences' in prop1.keys():
            plt.vlines(max_inds,corr[max_inds]-prop1['prominences'],corr[max_inds])
            plt.vlines(min_inds,corr[min_inds]+prop2['prominences'],corr[min_inds])
        plt.legend()
        plt.show()
    pmax = max_inds[None,:] - max_inds[:,None]
    pmin = min_inds[None,:] - min_inds[:,None]
    pmax = common_period(np.abs(pmax),tol=1)
    pmin = common_period(np.abs(pmin),tol=1)
    if debug:
        print('===================')
        print('maximum: %s'%max_inds)
        print('minimum: %s'%min_inds)
    if debug:
        print('===================')
        print('period for maximum:')
        print('  period   count  ')
        for p, c in zip(pmax[0],pmax[1]):
            print('   %d        %d'%(p,c))
        print('===================')
        print('period for minimum:')
        print('  period   count  ')
        for p, c in zip(pmin[0],pmin[1]):
            print('   %d        %d'%(p,c))
    maxc = pmax[1]
    pmax = pmax[0]
    minc = pmin[1]
    pmin = pmin[0]
    if len(pmax) == 0:
        if len(pmin) != 0:
            pmax = pmin[0]
            pmin = pmin[0]
        else:
            pmax = np.inf
            pmin = np.inf
    elif len(pmin) == 0:
            pmin = pmax[0]
            pmax = pmax[0]
    else:
        pmax = pmax[maxc == maxc[0]]
        pmax = np.sort(pmax)
        pmax = pmax[pmax < 1.5*pmax[0]]
        pmin = pmin[minc == minc[0]]
        pmin = np.sort(pmin)
        pmin = pmin[pmin < 1.5*pmin[0]]
        pcommon = list(set(pmin).intersection(pmax))
        if debug:
            print('===================')
            print('common period for maximum and minimum: %s'%pcommon)
        if len(pcommon) == 0:
            pmin = pmin[0]
            pmax = pmax[0]
        else:
            pmin = np.sort(pcommon)[0]
            pmax = np.sort(pcommon)[0]
    if pmax != np.inf:
        assert np.abs(pmax-pmin)<=1,'period for maximum is %s but for minimum is %s'%(pmax, pmin)
    else:
        warnings.warn('No period was found, set it as infinity!')
    if len(max_inds) != 0:
        start_inds, counts = np.unique(max_inds%pmax, return_counts = True)
        start_ind = start_inds[np.argsort(counts)][-1]
        period = pmax
    elif len(min_inds) != 0:
        warnings.warn('No maximum was detected, use minimum - period/2. to estimate!')
        start_inds, counts = np.unique(min_inds%pmin, return_counts = True)
        start_ind = start_inds[np.argsort(counts)][-1] - pmin/2.
        period = pmin
    else:
        warnings.warn('No maximum and minimum was detected, set start_ind as 0!')
        start_ind = 0.
        period = np.inf
    if debug:
        print('===================')
        print('start_inds  count')
        for s,c in zip(start_inds, counts):
            print('    %d         %d'%(s,c))
    if mode.strip() == 'valid':
        shift_x = x[0] - xwin[0] + start_ind*dx
    elif mode.strip() == 'full':
        shift_x = x[0] - xwin[0] - (len(window)-1)*dx + start_ind*dx
    elif mode.strip() == 'same':
        # left: int(len(xwin)/2) + 1
        # right: int(len(xwin)/2)
        shift_x = x[0] - xwin[0] - np.ceil((len(window) - 1)/2.)*dx + start_ind*dx
    else:
        raise 'Invalid mode %s'%mode.strip()
    if exchange:
        shift_x = -shift_x
    shift_x = shift_x%(period*dx)
    if shift_x > period*dx/2.:
        shift_x = shift_x - period*dx
    return shift_x, period*dx, pmax*dx, pmin*dx

def test():
    df = autogradientarr(autonp.cos, 4)
    df1 = []
    for i in range(5):
        df1 = df1 + [lambda x: df[i](1.*x)]
    return df1

if __name__ == '__main__':
    print('test!!!')
    '''
        test for sinc_fit
    '''
#    x = np.linspace(-3*np.pi,3*np.pi,1001)
#    w0 = np.abs(np.random.rand())*np.pi
#    polyr = [3,5,1,2]
#    polyi = [2,4,6,9]
#    y0 = SincPoly(w0,3*np.pi,0,polyr,imag=False)(x)+SincPoly(w0,3*np.pi,0,polyi,imag=False)(x)*1.J
#    print('=============================')
#    print(w0, 3*np.pi, polyr,polyi)
#    print('=============================')
#    poly1 = np.random.rand(4)*10
#    poly2 = np.random.rand(4)*10
#    print(poly1)
#    print(poly2)
#    print('=============================')
#    y = y0 + np.random.normal(scale = 0.03, size = y0.shape[0])
#    p = sinc_fit(x,y,3.0*np.pi,None,poly1,poly2,debug=True,max_iter=5)
#    print(p[0])
#    print(p[1])
    '''
        test for gaussian_fit
    '''
#    x = np.linspace(-100,5,10001)
#    y0 = gaussian(x, 2., 1., [2, 0, 3], normalize = False)
#    print(( 2., 1., 2, 0, 3))
#    y = y0 + np.random.normal(scale = 0.03, size = y0.shape[0])
#    p = gaussian_fit(x,y,2.2,0.8,[2.2, 0.2, 2.7],[9000,10001],debug = True)
#    plt.plot(x[9000:],y0[9000:],'.')
#    plt.plot(x[9000:],y[9000:],'*')
#    print(p[0])
#    plt.plot(x[9000:],gaussian(x[9000:],p[0][0],p[0][1],p[0][2:],False))
#    plt.show()
    '''
        test for gaussian
    '''
#    x = np.linspace(-15,18,2**10+1)
#    y = gaussian(x,0.5,1.5,[1,2,3,4],True)
#    print(romb(y, x[1]-x[0]))
#    plt.plot(x,y)
#    plt.show()
    '''
        test for factorial
    '''
#    import time
#    t0 = time.time()
#    print(factorial(20))
#    t1 = time.time()
#    print(sc.misc.factorial(20))
#    t2 = time.time()
#    print(t1 - t0)
#    print(t2 - t1)
    '''
        test for autogradient
    '''
#    x = np.linspace(-3*np.pi, 3*np.pi, 301)
#    y = -np.cos(x)
#    x = np.linspace(-1.2,1.2,301)
#    y = np.polyval([1,5,20,60,120,0],x)
#    def fun(x):
#        return autonp.sin(x)

#    dys = autogradientn(fun, 5)
#    dys = autogradientn(fun,3) 
#    dys = test()
#    print(dys)
#    plt.plot(x,y,'o')
#    plt.plot(x,dys(x),'r.')
#    plt.show()
#    clm = 'o^-:*'
#    for i in range(5):
#        print(dys[i](1.))
#        plt.plot(x,dys[i](x),'r'+clm[i])
#        plt.plot(x,dy1[i](x),'b'+clm[i])
#    plt.show()
    '''
        test for SincPoly
    '''
#    x = np.linspace(-4*np.pi,4*np.pi,50001)
#    n = 2
#    yfft,y = SincPoly(3.,4*np.pi,n,np.random.rand(n+1),return_all=True, imag = False)
#    for i in range(n + 1):
#        plt.figure(i)
#        yp = y[i](x)
#        print('======================')
#        print(x[yp==yp.max()])
#        print(x[yp==yp.min()])
#        yp = yp/yp.max()
#        plt.plot(x, yp)
#    plt.figure(n+1)
#    yf = yfft(x)
#    print('======================')
#    print(x[yf.real == yf.real.max()])
#    print(x[yf.real == yf.real.min()])
#    print(x[yf.imag == yf.imag.max()])
#    print(x[yf.imag == yf.imag.min()])
#    plt.plot(x,yf.real,'r-',label='real')
#    plt.plot(x,yf.imag,'b-',label='imag')
#    plt.ylim([plt.ylim()[0]*1.2,plt.ylim()[1]*1.2])
#    plt.legend()
#    plt.show()

    '''
        test for fft1d
    '''
    # multi-dimensional test
#    p = [1,0]
#    print(p[-2:])
#    x = np.linspace(-3*np.pi,3*np.pi,2001)
#    print(x[1] - x[2])
#    x += np.random.rand(x.shape[0])*0.2
#    y = [np.cos(x),np.cos(2*x)]
#    y = np.array(y).T
#    y = np.array([y,y])
#    y1 = np.cos(np.polyval(p[-2:],x))
#    k, yk, _ = fft1d(x,y,interp_kind='cubic',dk=0.01, debug=True,axis=1)
#    yk = np.transpose(yk, [1,0,2]).reshape([k.shape[0],-1])
#    for i in np.arange(yk.shape[1]):
#        yp = yk[:,i]
#        rmax = k[yp.real == yp.real.max()]
#        rmin = k[yp.real == yp.real.min()]
#        imax = k[yp.imag == yp.imag.max()]
#        imin = k[yp.imag == yp.imag.min()]
#        print('real maximum %s'%rmax)
#        print('real minimum %s'%rmin)
#        print('imag maximum %s'%imax)
#        print('imag minimum %s'%imin)
#    plt.figure(1)
#    plt.plot(k, yk.real, 'r-')
#    plt.figure(2)
#    plt.plot(k, yk.imag, 'b-')
#    plt.xlim([-20,20])
#    plt.show(block=False)
#    raw_input()

    # use sin
#    p = np.random.rand(4)
#    print('p:\n%s'%p)
#    x = np.linspace(-2*np.pi,2*np.pi,1001)
#    x += np.random.normal(scale=0.1, size = x.shape[0])
#    y = np.cos(x)*np.polyval(p,x)
#    k, yk, _ = fft1d(x,y,dk=0.01)
#    yk = yk[k!=0]
#    k = k[k!=0]
#    plt.plot(k, yk.real, 'r-')
#    plt.plot(k, yk.imag, 'b-')
#    ykc = SincPoly(1, 2.*np.pi, p = p)
#    plt.plot(k, ykc(k).real, 'r*')
#    plt.plot(k, ykc(k).imag, 'b*')
#    plt.show()

    # use gaussian
#    x = np.linspace(-5,5,1001) 
#    x += np.random.normal(loc=0,scale=10,size=x.shape[0])*1
#    a = 2.
#    y = [np.exp(-a**2*x**2)]
#    b = 3.
#    y += [np.exp(-b**2*x**2)]
#    k,yk,interpf = fft1d(x,y,dk=1.e-2)
#    yka = np.exp(-k**2/4./a**2)/2**0.5/a
#    ykb = np.exp(-k**2/4./b**2)/2**0.5/b
#    plt.figure()
#    plt.plot(k,yk[0].real,'r.',label='real')
#    plt.plot(k,yka,'r:',label='theory')
#    plt.plot(k,yk[0].imag,'b*',label='imag')
#    print('imag std: %s'%yk[0].imag.std())
#    print('real std %s'%(yk[0].real-yka).std())
#    print('real mean %s'%(yk[0].real-yka).mean())
#    plt.legend()
#    plt.figure()
#    plt.plot(k,yk[1].real,'r.',label='real')
#    plt.plot(k,ykb,'r:',label='theory')
#    plt.plot(k,yk[1].imag,'b*',label='imag')
#    print('imag std: %s'%yk[1].imag.std())
#    print('real std %s'%(yk[1].real-ykb).std())
#    print('real mean %s'%(yk[1].real-ykb).mean())
#    plt.legend()
#    if interpf is not None:
#        plt.figure()
#        plt.plot(x,y[0],'r.',label='data')
#        plt.plot(x,interpf(x)[0],'b*',label='interp')
#        plt.legend()
#        plt.figure()
#        plt.plot(x,y[1],'r.',label='data')
#        plt.plot(x,interpf(x)[1],'b*',label='interp')
#        plt.legend()
#    plt.show()

    '''
        test for correlation_shift
    '''
    xwin = np.linspace(0,8*np.pi,801)
    dx = xwin[1] - xwin[0]
    x = np.arange(0,1*np.pi,dx)
    y = np.sin(x) + np.random.normal(scale = .001, size = x.shape[0])
    window = np.sin(xwin)
    x0 = -13*dx
    xwin += x0
    a,b,c,d = correlation_shift(y, window,x,xwin,prominence_std=0,debug=True,mode='valid')
    print('+++++++++++++++++++++++++++++++')
    print(a,b,c,d)
    print(np.array([a,b,c,d])/dx)
    plt.plot(x-a,y)
    plt.plot(xwin, window)
    plt.show()
    























































