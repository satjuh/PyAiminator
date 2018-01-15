import numpy as np 


def check_bit_array(array):
    """
    Checks if a numpy array is bit-like (consisting of only two typ of
    values or less - zeros and ones.)

    :param array: numpy array
    :return: True if image is bit-like
             False if any other was found
    """
    for a in array:
        unique = np.unique(a)
        if len(unique) > 2:
            return False

    return True
