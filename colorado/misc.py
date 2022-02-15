import numpy as _np


def sphere(n):
    """return a sphare in a cubic volume of size n"""
    X, Y, Z = _np.mgrid[-1:1:n*1j, -1:1:n*1j, -1:1:n*1j]
    R2 = X**2+Y**2+Z**2
    vol = 1/_np.sqrt(R2)
    vol[vol < 1] = 0
    return vol
