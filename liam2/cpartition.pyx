# this file is based on code from pandas / pandas / src /
#
# Copyright (c) 2012 Federal Planning Bureau
# All rights reserved.
#
# Copyright (c) 2008-2011 AQR Capital Management, LLC
# All rights reserved.
#
# Copyright (c) 2011 Wes McKinney and pandas developers
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.
#
#     * Neither the name of the copyright holder nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from cpython cimport Py_INCREF, PyTuple_SET_ITEM, PyTuple_New
from cpython cimport PyObject

from libc.stdlib cimport malloc, free

cimport cython

cimport numpy as np
import numpy as np
from numpy cimport int8_t, int64_t, int32_t, uint32_t, ndarray


cdef extern from "khash.h":
    ctypedef uint32_t khint_t
    ctypedef khint_t khiter_t

    ctypedef struct kh_pymap_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        PyObject **keys
        Py_ssize_t *vals

    inline kh_pymap_t* kh_init_pymap()
    inline void kh_destroy_pymap(kh_pymap_t*)
    inline void kh_clear_pymap(kh_pymap_t*)
    inline khint_t kh_get_pymap(kh_pymap_t*, PyObject*)
    inline void kh_resize_pymap(kh_pymap_t*, khint_t)
    inline khint_t kh_put_pymap(kh_pymap_t*, PyObject*, int*)
    inline void kh_del_pymap(kh_pymap_t*, khint_t)

    bint kh_exist_pymap(kh_pymap_t*, khiter_t)

    ctypedef struct kh_pyset_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        PyObject **keys
        Py_ssize_t *vals

    inline kh_pyset_t* kh_init_pyset()
    inline void kh_destroy_pyset(kh_pyset_t*)
    inline void kh_clear_pyset(kh_pyset_t*)
    inline khint_t kh_get_pyset(kh_pyset_t*, PyObject*)
    inline void kh_resize_pyset(kh_pyset_t*, khint_t)
    inline khint_t kh_put_pyset(kh_pyset_t*, PyObject*, int*)
    inline void kh_del_pyset(kh_pyset_t*, khint_t)

    bint kh_exist_pyset(kh_pyset_t*, khiter_t)

    ctypedef char* kh_cstr_t

    ctypedef struct kh_str_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        kh_cstr_t *keys
        Py_ssize_t *vals

    inline kh_str_t* kh_init_str()
    inline void kh_destroy_str(kh_str_t*)
    inline void kh_clear_str(kh_str_t*)
    inline khint_t kh_get_str(kh_str_t*, kh_cstr_t)
    inline void kh_resize_str(kh_str_t*, khint_t)
    inline khint_t kh_put_str(kh_str_t*, kh_cstr_t, int*)
    inline void kh_del_str(kh_str_t*, khint_t)

    bint kh_exist_str(kh_str_t*, khiter_t)

    ctypedef struct kh_int64_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        int64_t *keys
        Py_ssize_t *vals

    inline kh_int64_t* kh_init_int64()
    inline void kh_destroy_int64(kh_int64_t*)
    inline void kh_clear_int64(kh_int64_t*)
    inline khint_t kh_get_int64(kh_int64_t*, int64_t)
    inline void kh_resize_int64(kh_int64_t*, khint_t)
    inline khint_t kh_put_int64(kh_int64_t*, int64_t, int*)
    inline void kh_del_int64(kh_int64_t*, khint_t)

    bint kh_exist_int64(kh_int64_t*, khiter_t)

    ctypedef struct kh_int32_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        int32_t *keys
        Py_ssize_t *vals

    inline kh_int32_t* kh_init_int32()
    inline void kh_destroy_int32(kh_int32_t*)
    inline void kh_clear_int32(kh_int32_t*)
    inline khint_t kh_get_int32(kh_int32_t*, int32_t)
    inline void kh_resize_int32(kh_int32_t*, khint_t)
    inline khint_t kh_put_int32(kh_int32_t*, int32_t, int*)
    inline void kh_del_int32(kh_int32_t*, khint_t)

    bint kh_exist_int32(kh_int32_t*, khiter_t)


@cython.wraparound(False)
@cython.boundscheck(False)
def filter_to_indices(ndarray[int8_t, cast=True] values):
    '''
    Create an array of indices where values is True.
    This is equivalent to: values.nonzero()[0]

    Arguments:
     * values: a vector of bool ((ndarray[bool8])

    Returns:
     * ndarray[int32] indices where values is True
    '''
    cdef:
        ndarray[int32_t] indices
        Py_ssize_t i, n = len(values)
        int32_t count_false, count
        int8_t val

    count_false, count = group_count_bool(values, True)
    indices = np.empty(count, dtype=np.int32)

    count = 0
    for i in range(n):
        val = values[i]
        if val:
            indices[count] = <int32_t>i
            count += 1

    return indices


@cython.wraparound(False)
@cython.boundscheck(False)
def group_count_bool(ndarray[int8_t, cast=True] values, object filter_value):
    '''
    Compute the number of True and False values in a vector of boolean values,
    with an optional filter.

    Arguments:
     * values: a vector of boolean values ((ndarray[bool8])
     * filter_value: either a single boolean (True) meaning there is no filter
                     or a vector of boolean values (ndarray[bool8]) which
                     filters out all values where the filter is False
    Returns:
     * count_false: number of occurrences of False
     * count_true: number of occurrences of True
    '''
    cdef:
        Py_ssize_t i, n = len(values)
        int32_t count_true = 0
        int32_t count_false
        int32_t count_values
        int8_t val
        ndarray[int8_t, cast=True] bool_filter
        int8_t keep_value

    if filter_value is True:
        for i in range(n):
            count_true += values[i]
        count_false = <int32_t>n - count_true
    else:
        assert isinstance(filter_value, np.ndarray) and (
                   filter_value.dtype.type is np.bool8)
        bool_filter = filter_value

        count_values = 0
        for i in range(n):
            keep_value = bool_filter[i]
            count_values += keep_value
            count_true += keep_value * values[i]

        count_false = count_values - count_true

    return count_false, count_true


@cython.wraparound(False)
@cython.boundscheck(False)
def _group_labels_int32(ndarray[int32_t] values, object filter_value):
    '''
    Compute the label vector, a "label->real value" dict and the number of
    occurrences of each label from a vector of int32 input values and an
    optional filter. See _group_labels for details.
    '''
    cdef:
        Py_ssize_t i, n = len(values)
        ndarray[int32_t] labels = np.empty(n, dtype=np.int32)
        ndarray[int32_t] counts = np.empty(n, dtype=np.int32)
        dict reverse = {}
        Py_ssize_t idx
        int32_t count = 0
        int ret
        int32_t val
        khiter_t k
        kh_int32_t *table
        ndarray[int8_t, cast=True] bool_filter
        int8_t keep_value

    table = kh_init_int32()
    kh_resize_int32(table, <khint_t>n)

    if filter_value is -1:
        # no explicit filter column but filter negative *values*
        for i in range(n):
            val = values[i]
            if val < 0:
                labels[i] = -1
                continue

            k = kh_get_int32(table, val)
            if k != table.n_buckets:
                idx = table.vals[k]
                labels[i] = <int32_t>idx
                counts[idx] = counts[idx] + 1
            else:
                k = kh_put_int32(table, val, &ret)
                if not ret:
                    kh_del_int32(table, k)
                table.vals[k] = count
                reverse[count] = val
                labels[i] = count
                counts[count] = 1
                count += 1
    elif filter_value is True:
        # no filter (take all values)
        for i in range(n):
            val = values[i]

            k = kh_get_int32(table, val)
            if k != table.n_buckets:
                idx = table.vals[k]
                labels[i] = <int32_t>idx
                counts[idx] = counts[idx] + 1
            else:
                k = kh_put_int32(table, val, &ret)
                if not ret:
                    kh_del_int32(table, k)
                table.vals[k] = count
                reverse[count] = val
                labels[i] = count
                counts[count] = 1
                count += 1
    else:
        # a filter column is provided
        assert isinstance(filter_value, np.ndarray) and (
                   filter_value.dtype.type is np.bool8)
        bool_filter = filter_value
        for i in range(n):
            keep_value = bool_filter[i]
            if keep_value:
                val = values[i]

                k = kh_get_int32(table, val)
                if k != table.n_buckets:
                    idx = table.vals[k]
                    labels[i] = <int32_t>idx
                    counts[idx] = counts[idx] + 1
                else:
                    k = kh_put_int32(table, val, &ret)
                    if not ret:
                        kh_del_int32(table, k)
                    table.vals[k] = count
                    reverse[count] = val
                    labels[i] = count
                    counts[count] = 1
                    count += 1
            else:
                labels[i] = -1

    kh_destroy_int32(table)

    return reverse, labels, counts[:count].copy()


@cython.wraparound(False)
@cython.boundscheck(False)
def _group_labels_int32_light(ndarray[int32_t] values, object filter_value):
    '''
    Compute the label vector and a "label->real value" dict from a vector of
    int32 input values and an optional filter.
    See _group_labels_light for details.
    '''
    cdef:
        Py_ssize_t i, n = len(values)
        ndarray[int32_t] labels = np.empty(n, dtype=np.int32)
        dict reverse = {}
        Py_ssize_t idx
        int32_t count = 0
        int ret
        int32_t val
        khiter_t k
        kh_int32_t *table
        ndarray[int8_t, cast=True] bool_filter
        int8_t keep_value

    table = kh_init_int32()
    kh_resize_int32(table, <khint_t>n)

    if filter_value is True:
        for i in range(n):
            val = values[i]
            k = kh_get_int32(table, val)
            if k != table.n_buckets:
                idx = table.vals[k]
                labels[i] = <int32_t>idx
            else:
                k = kh_put_int32(table, val, &ret)
                #XXX: the behavior on put failure seem weird:
                # table.vals[k] = count
                # should probably not happen in that case
                if not ret:
                    kh_del_int32(table, k)
                table.vals[k] = count
                reverse[count] = val
                labels[i] = count
                count += 1
    else:
        assert isinstance(filter_value, np.ndarray) and (
                   filter_value.dtype.type is np.bool8)
        bool_filter = filter_value
        for i in range(n):
            keep_value = bool_filter[i]
            if keep_value:
                val = values[i]
                k = kh_get_int32(table, val)
                if k != table.n_buckets:
                    idx = table.vals[k]
                    labels[i] = <int32_t>idx
                else:
                    k = kh_put_int32(table, val, &ret)
                    if not ret:
                        kh_del_int32(table, k)
                    table.vals[k] = count
                    reverse[count] = val
                    labels[i] = count
                    count += 1
            else:
                labels[i] = -1

    kh_destroy_int32(table)

    return reverse, labels


@cython.wraparound(False)
@cython.boundscheck(False)
def _group_labels_generic(ndarray[object] values, object filter_value):
    '''
    Compute the label vector, a "label->real value" dict and the number of
    occurrences of each label from a vector of input values of any type and an
    optional filter. See _group_labels for details.
    '''
    cdef:
        Py_ssize_t i, n = len(values)
        ndarray[int32_t] labels = np.empty(n, dtype=np.int32)
        ndarray[int32_t] counts = np.empty(n, dtype=np.int32)
        dict ids = {}, reverse = {}
        int32_t idx
        object val
        int32_t count = 0
        ndarray[int8_t, cast=True] bool_filter
        int8_t keep_value

    if filter_value is True:
        for i in range(n):
            val = values[i]

            # is NaN
            if val != val:
                labels[i] = -1
                continue

            # for large number of groups, not doing try: except: makes a big
            # difference
            if val in ids:
                idx = ids[val]
                labels[i] = idx
                counts[idx] = counts[idx] + 1
            else:
                ids[val] = count
                reverse[count] = val
                labels[i] = count
                counts[count] = 1
                count += 1
    else:
        assert isinstance(filter_value, np.ndarray) and (
                   filter_value.dtype.type is np.bool8)
        bool_filter = filter_value
        for i in range(n):
            keep_value = bool_filter[i]
            if keep_value:
                val = values[i]

                # is NaN
                if val != val:
                    labels[i] = -1
                    continue

                # for large number of groups, not doing try: except: makes a
                # big difference
                if val in ids:
                    idx = ids[val]
                    labels[i] = idx
                    counts[idx] = counts[idx] + 1
                else:
                    ids[val] = count
                    reverse[count] = val
                    labels[i] = count
                    counts[count] = 1
                    count += 1
            else:
                labels[i] = -1

    return reverse, labels, counts[:count].copy()


@cython.wraparound(False)
@cython.boundscheck(False)
def _group_labels_generic_light(ndarray[object] values, object filter_value):
    '''
    Compute the label vector and a "label->real value" dict from a vector of
    input values of any type and an optional filter.
    See _group_labels_light for details.
    '''
    cdef:
        Py_ssize_t i, n = len(values)
        ndarray[int32_t] labels = np.empty(n, dtype=np.int32)
        dict ids = {}, reverse = {}
        int32_t idx
        object val
        int32_t count = 0
        ndarray[int8_t, cast=True] bool_filter
        int8_t keep_value

    if filter_value is True:
        for i in range(n):
            val = values[i]

            # is NaN
            if val != val:
                labels[i] = -1
                continue

            # for large number of groups, not doing try: except: makes a big
            # difference
            if val in ids:
                idx = ids[val]
                labels[i] = idx
            else:
                ids[val] = count
                reverse[count] = val
                labels[i] = count
                count += 1
    else:
        assert isinstance(filter_value, np.ndarray) and (
                   filter_value.dtype.type is np.bool8)
        bool_filter = filter_value
        for i in range(n):
            keep_value = bool_filter[i]
            if keep_value:
                val = values[i]

                # is NaN
                if val != val:
                    labels[i] = -1
                    continue

                # for large number of groups, not doing try: except: makes a
                # big difference
                if val in ids:
                    idx = ids[val]
                    labels[i] = idx
                else:
                    ids[val] = count
                    reverse[count] = val
                    labels[i] = count
                    count += 1
            else:
                labels[i] = -1

    return reverse, labels


@cython.wraparound(False)
@cython.boundscheck(False)
def _group_labels(ndarray values, object filter_value):
    '''
    Compute the label vector, a "label->real value" dict and the number of
    occurrences of each label from a vector of input values of any type and an
    optional filter.

    Arguments:
     * values: vector (ndarray) of any Python object.
               bool8 and int32 have fast paths.
     * filter_value: can have any of the following values:
       - a single bool (True): meaning there is no filter/take all values
                               except NaNs
       - a single integer (-1): filter out all negative values (this is only
                                valid when values is an integer column:
                                int8 or int32).
       - a vector of bool (np.bool8): filter out all values where the filter
                                      is False

    Returns:
     * reverse: a dict {label: real value}
     * labels: ndarray[int32] containing the label for each value in the input.
               -1 for NaNs and filtered values
     * counts: ndarray[int32] with the number of occurrences for each label.
    '''
    cdef:
        ndarray[int32_t] counts
        ndarray[int32_t] labels
        int32_t count_false, count_true, n
        ndarray[int8_t, cast=True] bool_filter

    value_type = values.dtype.type

    if filter_value is -1:
        # group_labels_int32 is the only one of the group_labels_xxx to
        # implement filter_value == -1
        assert value_type in (np.int8, np.int32, np.intc), \
               "bad value type %s" % value_type

    if value_type is np.bool8:
        n = <int32_t>len(values)
        count_false, count_true = group_count_bool(values, filter_value)
        if count_false == 0 and count_true == 0:
            reverse = {}
            labels = np.zeros(n, dtype=np.int32)
            counts = np.zeros(0, dtype=np.int32)
        elif count_false == 0:
            reverse = {0: True}
            labels = np.zeros(n, dtype=np.int32)
            # count_true == n - num_missing
            # these lines are ugly but faster than
            # counts = np.array([count_false])
            counts = np.empty(1, dtype=np.int32)
            counts[0] = count_true
        elif count_true == 0:
            reverse = {0: False}
            labels = np.zeros(n, dtype=np.int32)
            # count_false == n - num_missing
            counts = np.empty(1, dtype=np.int32)
            counts[0] = count_false
        else:
            reverse = {0: False, 1: True}
            labels = values.astype(np.int32)
            counts = np.empty(2, dtype=np.int32)
            counts[0] = count_false
            counts[1] = count_true

        if filter_value is not True:
            assert (isinstance(filter_value, np.ndarray) and
                    filter_value.dtype.type is np.bool8)
            bool_filter = filter_value
            labels[~bool_filter] = -1
        return reverse, labels, counts
    elif value_type is np.int8:
        return _group_labels_int32(values.astype(np.int32), filter_value)
    elif value_type in (np.int32, np.intc):
        return _group_labels_int32(values, filter_value)
    else:
        return _group_labels_generic(values.astype(object), filter_value)


#TODO: add special cases for int64
#TODO: use int32 and int64 functions for types coded on 4 and 8 bytes
# using for example "f32.view(np.int32)" for float32
@cython.wraparound(False)
@cython.boundscheck(False)
def _group_labels_light(ndarray values, object filter_value):
    '''
    Compute the label vector and a "label->real value" dict from a vector of
    input values of any type and an optional filter.

    Arguments:
     * values: vector (ndarray) of any Python object.
               bool8 and int32 have fast paths.
     * filter_value: can be either:
       - a single bool (True): meaning there is no filter/take all values
                               except NaNs
       - a vector of bool (np.bool8): filter out all values where the filter
                                      is False

    Returns:
     * reverse: a dict {label: real value}
     * labels: ndarray[int32] containing the label for each value in the input.
               -1 for NaNs and filtered values
    '''
    cdef:
        ndarray[int32_t] labels
        ndarray[int8_t, cast=True] bool_filter

    value_type = values.dtype.type
    if value_type is np.bool8:
        # In this case, we could cast to int8 instead but the minor performance
        # improvement (86ms -> 83ms) is not worth having a "varying" return
        # type.
        labels = values.astype(np.int32)
        if filter_value is not True:
            assert (isinstance(filter_value, np.ndarray) and
                    filter_value.dtype.type is np.bool8)
            bool_filter = filter_value
            labels[~bool_filter] = -1

        # It is not really a problem if there is not any True or False
        # label because group_labels_light is only an internal function.
        # However, if we someday need it to be strict about having a
        # minimal "reverse" dictionary, we have to make sure that the
        # labels are adapted too (ie no individual label value is
        # >= len(reverse)!
        return {0: False, 1: True}, labels
    elif value_type is np.int8:
        return _group_labels_int32_light(values.astype(np.int32), filter_value)
    elif value_type is np.int32:
        return _group_labels_int32_light(values, filter_value)
    else:
        return _group_labels_generic_light(values.astype(object), filter_value)


@cython.wraparound(False)
@cython.boundscheck(False)
def group_indices_nd(list columns, object filter_value):
    '''
    For each combination of values in columns, return the list of indices that
    match this combination.

    Arguments:
     * columns: a list of vectors (ndarray) of any type, though only bool8
                and int32 have fast paths.
     * filter_value: can be either:
       - a single bool (True): meaning there is no filter/take all values
                               except NaNs
       - a vector of bool (np.bool8): filter out all values where the filter
                                      is False
    Returns:
      {(dim1_val0, ..., dimN_val0): indices (ndarray[int32]) for that group,
       (dim1_val1, ..., dimN_val0): indices (ndarray[int32]) for that group,
       ...}
    '''
    cdef:
        Py_ssize_t i, j, ndim, n, totalcount
        ndarray[int32_t] counts, arr, seen
        ndarray values, labels, combined_labels
        ndarray[int32_t] labels32
        int32_t count
        int32_t loc
        int32_t packed_id
        int32_t divisor
        dict ids = {}
        list dim_id_maps = []
        int32_t label
        tuple tup
        dict dim_id_map
        int32_t dim_id
        object dim_val
        ndarray[int8_t, cast=True] bool_filter
        int8_t keep_value

    ndim = len(columns)
    assert ndim > 0
    assert filter_value is True or \
           (isinstance(filter_value, np.ndarray) and
            filter_value.dtype.type is np.bool8)

    values = columns[0]
    if ndim > 1:
        ids, labels = _group_labels_light(values, filter_value)
        totalcount = len(ids)
        # The tricky case is when count (len(ids)) is 1 but in that case, the
        # labels should all be 0, so we do fine.
        combined_labels = labels
        dim_id_maps.append(ids)

        for i in range(1, ndim):
            values = columns[i]
            ids, labels = _group_labels_light(values, filter_value)

            # We need the np.where to handle the case where there is a
            # missing value (eg. NaN) in an earlier dimension (eg. dim0)
            # but not on a subsequent dimension. Without the where(), the
            # combined label would be a positive value when it should be
            # negative.
            combined_labels = np.where(combined_labels < 0,
                                       -1,
                                       labels * totalcount + combined_labels)
            totalcount *= len(ids)
            if totalcount > 2 ** 31:
                raise Exception("too many combination of values: the "
                                "combined labels (32bits) overflowed")
            dim_id_maps.append(ids)

        # combined_labels can be < 0 for filtered values or NaNs
        ids, labels32, counts = _group_labels(combined_labels, -1)
    else:
        ids, labels32, counts = _group_labels(values, filter_value)

    # allocate a list of pointers: for each "group", we will store the direct
    # pointer to the "data" of its corresponding indices ndarray
    cdef int32_t **vecs = <int32_t **> malloc(len(ids) * sizeof(int32_t*))

    # preallocate all indices ndarrays, link to them in the "result" dict and
    # store their direct "data" pointers in "vecs"
    # result = {(dim1_val0, dim2_val0, dim3_val0): empty ndarray(int32),
    #           (dim1_val1, dim2_val0, dim3_val0): empty ndarray(int32),
    #           (...): ...}
    result = {}
    if ndim > 1:
        # for each group
        for i in range(len(counts)):
            # allocate a new array
            arr = np.empty(counts[i], dtype=np.int32)

            # "unpack" id (= combined_label) to a tuple (dim1_val0, ...)
            packed_id = ids[i]
            tup = PyTuple_New(ndim)
            divisor = 1
            for j in range(ndim):
                dim_id_map = dim_id_maps[j]
                count = <int32_t>len(dim_id_map)
                dim_id = (packed_id / divisor) % count
                dim_val = dim_id_map[dim_id]
                Py_INCREF(dim_val)
                PyTuple_SET_ITEM(tup, j, dim_val)
                divisor *= count

            # store the array in the result map
            result[tup] = arr

            # store pointer to each array
            vecs[i] = <int32_t *> arr.data
    else:
        for i in range(len(counts)):
            arr = np.empty(counts[i], dtype=np.int32)
            result[ids[i]] = arr
            # pointers to each array
            vecs[i] = <int32_t *> arr.data

    # keep track of how many indices we have already written in each result
    # vector
    seen = np.zeros_like(counts)

    # populate the array of indices for each group
    # labels32 can have -1 if any(values == NaN)
    n = len(values)
    if filter_value is True:
        for i in range(n):
            label = labels32[i]
            if label >= 0:
                loc = seen[label]
                vecs[label][loc] = <int32_t>i
                seen[label] = loc + 1
    else:
        bool_filter = filter_value
        assert len(bool_filter) == n
        for i in range(n):
            keep_value = bool_filter[i]
            if keep_value:
                label = labels32[i]
                if label >= 0:
                    loc = seen[label]
                    vecs[label][loc] = <int32_t>i
                    seen[label] = loc + 1

    free(vecs)
    return result
