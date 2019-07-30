import numpy as np

def is_numeric_array(array):
    """Checks if the dtype of the array is numeric.
    Booleans, unsigned integer, signed integer, floats and complex are considered numeric. 
    Parameters
    ----------
    array : `numpy.ndarray`-like. The array to check.

    Returns
    -------
    is_numeric : `bool`. True if it is a recognized numerical and False if object or string.
    """
    numerical_dtype_kinds = {'b', # boolean
                             'u', # unsigned integer
                             'i', # signed integer
                             'f', # floats
                             'c'} # complex
    return np.asarray(array).dtype.kind in numerical_dtype_kinds

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

def uneven_arr_split(array, section_size = 1, axis = 0):
    '''split a numeric array into several part, a reverse to np.concantenate()
    Parameters:
    ----------
    array: the array to split, must only contain numeric elements
    section_size: int or array
                  if int, this function is equivalent to np.array_split
                  if array, the array will be split into several parts with length according to elements in section _size.
    axis: alone which axis the split is done
    Return:
    ----------
    new_arrs: list, the split arrays
    '''
    assert is_numeric_array(array), 'Elements in the array must be numerical, i.e. b, u, i, f, c'
    if type(section_size) is int:
        return np.array_split(array, section_size, axis = axis)
    array = np.array(array)
    tax, rax = axes_trans(array, axis)
    array = np.transpose(array, tax)
    if sum(section_size) != array.shape[0]:
        raise Exception('the section_size %d is not equal length of the array %d'%(sum(section_size), len(array)))

    new_arrs = []
    section_size = np.cumsum(np.append(0,section_size))
    for i in range(section_size.shape[0] - 1):
        new_arrs += [np.transpose(array[section_size[i]:section_size[i+1]], rax)]
    return new_arrs













