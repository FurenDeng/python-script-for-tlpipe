import numpy as np
from scipy.linalg import eigh

def random_realize(mat, random_gen, outshape = None):
    '''
    Random realization, can be used to decompose Cl into alm.
    Decompose square mat A(m*m) into B of any shape (m,n) that satisfies A = np.dot(B, B.conj().T)
    The B is independent of matrix decomposition method.

    Parameter:
    ==========
    mat: (m*m) array, square matrix
    random_gen: function, random_gen(a1, a2) should generate random number with shape (a1, a2). the mean of random number should be 0, and std should be 1.
    outshape: (m, n), tuple with length 2. the shape of output matrix. outshape[0] should be m. if None, will be the same as mat. default None.

    Return:
    ==========
    decomp_mat: (m, n) array, the result of random realization
    '''
    mat = np.array(mat)
    if outshape is None:
        outshape = mat.shape
    elif outshape[0] != mat.shape[0]:
        raise Exception('First dimension of the output matrix %d is not the same as the input matrix %d!'%(outshape[0], mat.shape[0]))
    w,v = eigh(mat)
    w = np.array(w, dtype = np.complex64)
    v = v*w[None,:]**0.5
    ran = random_gen(*outshape)/1./outshape[1]**0.5
    return np.dot(v,ran)
if __name__=='__main__':
    ndim = 3
    nran = 50000
    a = np.random.rand(ndim,ndim) + np.random.rand(ndim,ndim)*1.J
    A = np.dot(a.conj().T,a)
    def gen1(a1,a2):
        return (np.random.rand(a1, a2)-0.5)*2.*3.**0.5
    def gen2(a1,a2):
        return np.random.normal(size = (a1, a2))
    def gen3(a1,a2):
        return np.random.normal(scale = 1/2.**0.5, size = (a1, a2)) + np.random.normal(scale = 1/2.**0.5, size = (a1, a2))*1.J
    print(A)
    dep = random_realize(A,gen1,[ndim,nran])
    print((np.dot(dep, dep.conj().T)-A)/np.abs(A))
    dep = random_realize(A,gen2,[ndim,nran])
    print((np.dot(dep, dep.conj().T)-A)/np.abs(A))
    dep = random_realize(A,gen3,[ndim,nran])
    print((np.dot(dep, dep.conj().T)-A)/np.abs(A))
