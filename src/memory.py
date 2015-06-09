from __future__ import division, print_function

import math
from types import BuiltinFunctionType, FunctionType


from utils import prod


class Manager(object):
    def __init__(self, cache):
        self.cache = cache

    def safe_call(self, nbytes, func, *args, **kwargs):
        # TODO: check & free mem beforehand (use psutil)
        try:
            array = func(*args, **kwargs)
        except MemoryError:
            self.free(nbytes)
            array = func(*args, **kwargs)
            print("succeeded!")
        # make sure the given size was correct
        assert array.nbytes == nbytes
        return array

    def decorate(self, sizefunc, func):
        def decorated(*args, **kwargs):
            return self.safe_call(sizefunc(*args, **kwargs),
                                  func, *args, **kwargs)
        decorated.__name__ = func.__name__
        return decorated

    def free(self, nbytes):
        print("mem is full, puring %d elements from cache" % len(self.cache))
        self.cache.clear()


class ManagedModule(object):
    def __init__(self, module, manager, sizefuncs):
        self.module = module
        self.manager = manager
        self.sizefuncs = sizefuncs

    def __getattr__(self, key):
        func = getattr(self.module, key)
        assert isinstance(func, (BuiltinFunctionType, FunctionType))
        return self.manager.decorate(self.sizefuncs[key], func)

if __name__ == '__main__':
    import numpy as np

    def generic(shape, dtype=float, *_, **__):
        if not isinstance(shape, tuple):
            shape = (shape,)
        dt = np.dtype(dtype)
        return prod(shape) * dt.itemsize

    def like(a, dtype=None, *_, **__):
        if dtype is None:
            return a.nbytes
        else:
            return generic(a.shape, dtype)

    def arange(start, stop=None, step=None, dtype=None):
        if dtype is None:
            dtype = type(start)
        if stop is None:
            start, stop = 0, start
        if step is None:
            step = 1
        return generic(int(math.ceil((stop - start) / step)), dtype)

    def aggregate_shape(a, axis=None, out=None):
        if out is not None:
            return 0
        elif axis is None:
            return 1
        else:
            if not isinstance(axis, tuple):
                axis = (axis,)
            tokill = set(axis)
            return tuple(size for i, size in enumerate(a.shape)
                         if i not in tokill)

    def make_aggregate(dt=None):
        def aggregate_func(a, axis=None, out=None):
            if dt is None:
                dt = a.dtype
            return generic(aggregate_shape(a, axis, out), dt)
        return aggregate_func

    cache = {}
    manager = Manager(cache)
    mnp = ManagedModule(np, manager, {
        'empty': generic,
        'zeros': generic,
        'ones': generic,
        'arange': arange,
        'empty_like': like,
        'copy': like,
        'sort': like,
        'all': make_aggregate(bool),
        'any': make_aggregate(bool),
        'sum': make_aggregate(),
        # TODO: array slices et al via ndarray subclass
        # TODO: array, asarray
        # agg: cumsum, nansum, min, nanmin, nanmax, std, median, percentile,
        # argsort, bincount,
        # insert, delete, concatenate, repeat
        # intersect1d, union1d, setdiff1d, repeat,
        # isnan, isfinite, clip, round,
        # where
        # random.xxx, unique,
    }) #, ignored=set('inexact,issubdtype')

    for i in range(50):
        print(i, end=' ')
        new = mnp.arange(1e7)
        print("ok")
        cache[i] = new