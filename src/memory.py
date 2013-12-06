from __future__ import division

import math

import numpy as np

from utils import prod


class Manager(object):
    def __init__(self):
        pass

    def allocate(self, nbytes, func, *args, **kwargs):
        try:
            array = func(*args, **kwargs)
        except MemoryError:
            self.free(nbytes)
            array = func(*args, **kwargs)
        assert array.nbytes == nbytes
        return array

    def empty(self, shape, dtype=float, order='C'):
        if not isinstance(shape, tuple):
            shape = (shape,)
        dt = np.dtype(dtype)
        nbytes = prod(shape) * dt.itemsize
        return self.allocate(nbytes, np.empty, shape, dt, order)

    def zeros(self, shape, dtype=float, order='C'):
        array = self.empty(shape, dtype, order)
        array[:] = 0
        return array

    def ones(self, shape, dtype=float, order='C'):
        array = self.empty(shape, dtype, order)
        array[:] = 1
        return array

    def arange(self, start, stop=None, step=None, dtype=None):
        if dtype is None:
            dtype = type(start)
        dt = np.dtype(dtype)
        if stop is None:
            start, stop = 0, start
        if step is None:
            step = 1
        nbytes = int(math.ceil((stop - start) / step)) * dt.itemsize
        return self.allocate(nbytes, np.arange, start, stop, step)

    def free(self, nbytes):
        pass

