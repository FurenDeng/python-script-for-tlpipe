import h5py as h5
import numpy as np
from scipy.stats import rv_continuous
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

#class random_gen(rv_continuous):
#    def __init__(self, func, *args, **kwargs):
#        def pdf_func(x):
#            return func(x, *args, **kwargs)
#        super(self.__class__, self).__init__()
#        self._pdf = pdf_func
#def gaussian(x, A, mean, sigma):
#    return A*np.exp(-(x - mean)**2/2./sigma**2)
#gaussian_ran = random_gen(gaussian, A = 1., mean = 0., sigma = 1.)
#def gauss(x):
#    return gaussian(x, 1., 0., 1.)
#gauss_ran = rv_continuous()
#gauss_ran._pdf = gauss
#plt.hist(gaussian_ran.rvs(size = 1000), bins = np.linspace(-3,0,21))
#plt.show()

def gaussian(x, A, mean, sigma):
    return A*np.exp(-(x - mean)**2/2./sigma**2)
def pdf_rvs(func, xlim = None, positive = False, truncate = 1.e-15, precision = 1.e-3, **kwargs):
    '''
    Calculate reverse function of cpf when pdf is given.
    
    Parameter:
    ==========
    func: callable, the pdf, should ->0 when abs(x)->inf, and should always > 0
    xlim: (2,) tuple, optional, define the section in which the pdf will be integrated, will be searched using truncate if not set
    positive: bool, optional, if True, only use x > precision
    truncate: truncate pdf integration if func(x) < truncate
    precision: the step size of integration
    **kwargs: kwargs that will be passed to func

    Return:
    ========
    rvs: callable, reverse function of cpf
    '''
    if xlim is None:
        for i in np.arange(50):
            xlim = np.exp(i)
            if positive:
                if func(xlim, **kwargs) < truncate:
                    break
            else:
                if func(xlim, **kwargs) < truncate and func(-xlim, **kwargs) < truncate:
                    break
        else:
            raise Exception('The function can not converge or converge very slow!')
        if positive:
            xlim = [precision, xlim]
        else:
            xlim = [-xlim, xlim]
    xarr = np.arange(xlim[0], xlim[1], precision)
    yarr = func(xarr, **kwargs)
    cumyarr = cumtrapz(yarr,xarr)
    cumyarr = np.append(0, cumyarr)/cumyarr[-1]
    cumyarr[-1] = 1.
    rvs = interp1d(cumyarr, xarr)
    return rvs
def random_gen(func, *args, **kwargs):
    '''
    Generate random number for given pdf.

    Parameter:
    ==========
    func: callable, the pdf, should ->0 when abs(x)->inf, and should always > 0
    args: a1, a2, a3 ..., int, define the shape of output array
    kwargs: kwargs that will be passed to func and pdf_rvs
    '''
    return pdf_rvs(func, **kwargs)(np.random.rand(np.prod(args))).reshape(args)
if __name__ == '__main__':
    rvs = pdf_rvs(lambda x: np.ones_like(x), xlim = [0,1], truncate = 1.e-20)#, A = 1., mean = 0., sigma = 1.)
#    rvs = pdf_rvs(lambda x: x**-4)
#    plt.hist(rvs(np.random.rand(100000)))
    mat = random_gen(gaussian, 300, 400, 500, truncate = 1.e-10, precision = 1.e-4, A = 10, mean = 10, sigma = 2.)
    plt.hist(mat.flatten())
    print(mat.shape)
    plt.show()
