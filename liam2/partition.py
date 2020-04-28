# encoding: utf-8
from __future__ import absolute_import, division, print_function

from itertools import product

import numpy as np
import larray as la

try:
    from liam2.cpartition import filter_to_indices
except ImportError:
    def filter_to_indices(filter_value):
        return filter_value.nonzero()[0]

try:
    from liam2.cpartition import group_indices_nd

    def partition_nd(columns, filter_value, possible_values):
        columns = [col.data if isinstance(col, la.Array) else col
                   for col in columns]
        filter_value = filter_value.data if isinstance(filter_value, la.Array) else filter_value
        ndim = len(columns)
        assert ndim > 0
        assert all(isinstance(c, np.ndarray) for c in columns), \
            "not all columns are ndarrays: " + \
            ', '.join(str(type(c)) for c in columns)
        # note that since we don't iterate through the columns many times,
        # it's not worth it to copy non contiguous columns in this version
        d = group_indices_nd(columns, filter_value)

        if possible_values is None:
            combined_labels_present = d.keys()
            if ndim > 1:
                possible_values = [np.unique(col) for col in zip(*combined_labels_present)]
            else:
                possible_values = [np.array(sorted(combined_labels_present))]
            assert len(possible_values) == ndim

        if ndim > 1:
            all_combined_values = product(*possible_values)
        else:
            all_combined_values = possible_values[0]
        empty_list = np.empty(0, dtype=int)

        # XXX: It would be nice to print a warning if d contains keys not in
        # combined_values but that might be hard to implement efficiently in the python
        # version (and I am not eager to diverge too much).
        return [d.get(pv, empty_list) for pv in all_combined_values], possible_values
except ImportError:
    group_indices_nd = None

    # TODO: make possible_values a list of combinations of value. In some cases,
    # (eg GroupBy), we are not interested in all possible combinations.
    def partition_nd(columns, filter_value, possible_values=None):
        """
        * columns is a list of columns containing the data to be partitioned
        * filter_value is a vector of booleans which selects individuals
        * possible_values is an matrix with N vectors containing the possible
          values for each column
        * returns a 1d array of lists of indices
        """
        # make a copy of non contiguous columns. It is only worth it when the
        # number of possible values for that column is large enough to
        # compensate for the cost of the copy, and it is usually the case.
        # XXX: we might want to be more precise about this.
        # 1e5 arrays
        # * not aligned (nor contiguous): always worth it
        # * aligned but not contiguous: never worth it
        # 1e6 arrays
        # * not aligned (nor contiguous): worth it from 6 values
        # * aligned but not contiguous: worth it from 12 values
        contiguous_columns = []
        for col in columns:
            if isinstance(col, np.ndarray) and col.shape:
                if not col.flags.contiguous:
                    col = col.copy()
            else:
                col = [col]
            contiguous_columns.append(col)
        columns = contiguous_columns

        if possible_values is None:
            possible_values = [np.unique(col) for col in columns]

        size = tuple([len(colvalues) for colvalues in possible_values])

        result = []

        # for each combination of i, j, k:
        for idx in np.ndindex(*size):
            # local_filter = filter & (data0 == values0[i])
            #                       & (data1 == values1[j])
            # it is a bit faster to do: v = expr; v &= b
            # than
            # v = copy(b); v &= expr
            parts = list(zip(idx, possible_values, columns))
            if parts:
                first_i, first_colvalues, first_coldata = parts[0]
                local_filter = first_coldata == first_colvalues[first_i]
                for i, colvalues, coldata in parts[1:]:
                    local_filter &= coldata == colvalues[i]
                local_filter &= filter_value
            else:
                # filter_value can be a simple boolean, in that case, we
                # get a 0-d array.
                local_filter = np.copy(filter_value)

            if isinstance(local_filter, np.ndarray) and local_filter.shape:
                group_indices = filter_to_indices(local_filter)
            elif not local_filter:
                # local_filter is False
                group_indices = np.empty(0, dtype=int)
            else:
                # local_filter is True
                assert local_filter
                group_indices = np.arange(len(columns[0]))
            result.append(group_indices)
        return result, possible_values

        # pure-python version. It is 10x slower than the NumPy version above
        # but it might be a better starting point to translate to C,
        # especially given that the possible_values are usually sorted (we
        # could sort them too), so we could use some bisect algorithm to find
        # which category it belongs to. python built-in bisect is faster
        # (average time on all indexes) than list.index() starting from ~20
        # elements. We usually have even less elements than that though :(.
        # Strangely bisect on a list is 2x faster than np.searchsorted on an
        # array, even with large array sizes (10^6).
#        fill_with_empty_list = np.frompyfunc(lambda _: [], 1, 1)
#        fill_with_empty_list(result, result)
#
#        for idx, row in enumerate(zip(*columns)):
#            # returns a tuple with the position of the group this row belongs
#            # to. eg. (0, 1, 5)
#            # XXX: this uses strict equality, partitioning using
#            # inequalities might be useful in some cases
#            if filter[idx]:
#                try:
#                    # pos = tuple([values_i.index(vi) for vi, values_i
#                    pos = tuple([np.searchsorted(values_i, vi)
#                                 for vi, values_i
#                                 in zip(row, possible_values)])
#                    result[pos].append(idx)
#                except ValueError:
#                    # XXX: issue a warning?
#                    pass
#        for idx in np.ndindex(*size):
#            result[idx] = np.array(result[idx])
#        return np.ravel(result)
