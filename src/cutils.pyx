cimport numpy as np
import numpy as np
from numpy cimport ndarray


def fromiter(iterable, dtype, Py_ssize_t count=-1):
    cdef ndarray buf
    cdef Py_ssize_t i
    cdef object e

    if count == -1:
        return np.fromiter(iterable, dtype)
    else:
        buf = np.empty(count, dtype=dtype)
        i = 0
        for e in iterable:
            buf[i] = e
            i += 1
            if i == count:
                break
        if i < count:
            raise ValueError("iterator too short")
        return buf
