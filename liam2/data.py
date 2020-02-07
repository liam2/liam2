# encoding: utf-8
from __future__ import absolute_import, division, print_function

import time

import tables
import numpy as np
import larray as la

from liam2.compat import basestring
from liam2 import config
from liam2.partition import filter_to_indices
from liam2.expr import normalize_type, get_default_value, get_default_array, get_default_vector, gettype
from liam2.utils import loop_wh_progress, time2str, safe_put, timed, MB
from liam2.importer import load_def, stream_to_array, array_to_disk_array


def anyarray_to_disk(node, name, array):
    if array.dtype.names is None:
        array_to_disk_array(node, name, array, title=name)
    else:
        # noinspection PyProtectedMember
        h5file = node._v_file
        table = h5file.create_table(node, name, array.dtype, title=name)
        table.append(array)
        table.flush()


def append_carray_to_table(array, table, numlines=None, buffersize=10 * MB):
    """
    Parameters
    ----------
    array : structured-array-like
        must be indexable by field name and each field value must be array-like.
        must contain at least all table fields (but can contain more).
    table : table-like
        Any object with .append(np.ndarray[structured_dtype]) and .flush() methods should work. tables.Table.
    numlines : int, optional
        Number of lines to append. Defaults to all lines.
    buffersize : int, optional
        Maximum buffer size. Defaults to 10 Mb.
    """
    dtype = table.dtype
    max_buffer_rows = buffersize // dtype.itemsize
    if numlines is None:
        numlines = len(array)
    buffer_rows = min(numlines, max_buffer_rows)
    chunk = np.empty(buffer_rows, dtype=dtype)
    start, stop = 0, buffer_rows
    while numlines > 0:
        buffer_rows = min(numlines, max_buffer_rows)
        if buffer_rows < len(chunk):
            # last chunk is smaller
            chunk = np.empty(buffer_rows, dtype=dtype)
        for fieldname in dtype.names:
            field_value = array[fieldname]
            chunk[fieldname] = field_value[start:stop]
        table.append(chunk)
        # TODO: try flushing after each chunk, this should reduce memory
        # use on large models, and (hopefully) should not be much slower
        # given our chunks are rather large
        # >>> on our 300k sample, it does not seem to make any difference
        #     either way. I'd like to test this on the 2000k sample, but
        #     that will have to wait for 0.8
        numlines -= buffer_rows
        start += buffer_rows
        stop += buffer_rows
    table.flush()


class ColumnArray(object):
    def __init__(self, array=None):
        if array is not None:
            columns = {}
            if isinstance(array, (np.ndarray, ColumnArray)):
                self.dtype = array.dtype
                for name in array.dtype.names:
                    columns[name] = array[name].copy()
                self.columns = columns
            elif isinstance(array, list):
                # list of (name, column) pairs
                self.dtype = np.dtype([(name, column.dtype)
                                       for name, column in array])
                for name, column in array:
                    columns[name] = column
                self.columns = columns
            else:
                raise TypeError('invalid array', array)
        else:
            self.dtype = None
            self.columns = {}

    def __getitem__(self, key):
        if isinstance(key, basestring):
            return self.columns[key]
        else:
            # int, slice, ndarray
            ca = ColumnArray()
            ca.columns = {colname: colvalue[key] for colname, colvalue in self.columns.items()}
            ca.dtype = self.dtype
            return ca

    def __setitem__(self, key, value):
        """does not copy value except if a type conversion is necessary"""

        if isinstance(key, basestring):
            length = len(self)
            if isinstance(value, la.Array) and value.shape:
                raise TypeError("la.Array not supported in ColumnArray, you should use LColumnArray instead")
            if isinstance(value, np.ndarray) and value.shape:
                if len(value) != length:
                    raise ValueError("could not broadcast input array from shape ({}) into shape ({})"
                                     .format(len(value), length))
                column = value
            else:
                # expand scalars (like ndarray does) so that we don't have to
                # check isinstance(x, ndarray) and x.shape everywhere
                # TODO: create ConstantArray(length, value, dtype)
                column = np.full(length, value, dtype=gettype(value))

            if key in self.columns:
                # converting to existing dtype
                if column.dtype != self.dtype[key]:
                    column = column.astype(self.dtype[key])
                self.columns[key] = column
            else:
                # adding a new column so we need to update the dtype
                self.columns[key] = column
                self._update_dtype()

#            ids = {}
#            for k, v in self.columns.items():
#                ids.setdefault(id(v), set()).add(k)
#            dupes = [v for k, v in ids.items() if len(v) > 1]
#            if dupes:
#                print "aliases", dupes
        else:
            # int, slice, ndarray
            for name, column in self.columns.items():
                column[key] = value[name]

    def put(self, indices, values, mode='raise'):
        for name, column in self.columns.items():
            column.put(indices, values[name], mode)

    @property
    def nbytes(self):
        return sum(v.nbytes for v in self.columns.values())

    def __delitem__(self, key):
        del self.columns[key]
        self._update_dtype()

    def _update_dtype(self):
        # handle fields already present (iterate over old dtype to preserve
        # order)
        if self.dtype is not None:
            old_fields = self.dtype.names
            fields = [(name, self.dtype[name])
                      for name in old_fields
                      if name in self.columns]
        else:
            old_fields = []
            fields = []
        # add new fields (not already handled)
        old_fields = set(old_fields)
        fields += [(name, column.dtype)
                   for name, column in self.columns.items()
                   if name not in old_fields]
        self.dtype = np.dtype(fields)

    def __len__(self):
        if len(self.columns):
            anycol = next(iter(self.columns.values()))
            return len(anycol)
        else:
            return 0

    # this is an inplace method so the old memory for each column can be freed before all columns have been changed
    def keep(self, key):
        """key can be either a vector of int indices or boolean filter"""

        # using gc.collect() after each column update frees a bit of memory
        # but slows things down significantly.
        for name, column in self.columns.items():
            self.columns[name] = column[key]

    # this is an inplace method so the old memory for each column can be freed before all columns have been changed
    def append(self, array):
        assert array.dtype == self.dtype, (array.dtype, self.dtype)
        # using gc.collect() after each column update frees a bit of memory
        # but slows things down significantly.
        for name, column in self.columns.items():
            self.columns[name] = np.concatenate((column, array[name]))

    def append_to_table(self, table, buffersize=10 * MB):
        append_carray_to_table(self, table, buffersize=buffersize)

    @classmethod
    def empty(cls, length, dtype):
        ca = cls()
        for name in dtype.names:
            ca.columns[name] = np.empty(length, dtype[name])
        ca.dtype = dtype
        return ca

    @classmethod
    def from_table(cls, table, start=0, stop=None, buffersize=10 * MB):
        # reading a table one column at a time is very slow, this is why this
        # function is even necessary
        if stop is None:
            stop = len(table)
        dtype = table.dtype
        max_buffer_rows = buffersize // dtype.itemsize
        numlines = stop - start
        ca = cls.empty(numlines, dtype)
        buffer_rows = min(numlines, max_buffer_rows)
        chunk = np.empty(buffer_rows, dtype=dtype)
        array_start = 0
        table_start = start
        while numlines > 0:
            buffer_rows = min(numlines, max_buffer_rows)
            if buffer_rows < len(chunk):
                # last chunk is smaller
                chunk = np.empty(buffer_rows, dtype=dtype)
            table.read(table_start, table_start + buffer_rows, out=chunk)
            ca[array_start:array_start + buffer_rows] = chunk
            table_start += buffer_rows
            array_start += buffer_rows
            numlines -= buffer_rows
        return ca

    @classmethod
    def from_table_coords(cls, table, indices, buffersize=10 * MB):
        dtype = table.dtype
        max_buffer_rows = buffersize // dtype.itemsize
        numlines = len(indices)
        ca = cls.empty(numlines, dtype)
        buffer_rows = min(numlines, max_buffer_rows)
        # chunk = np.empty(buffer_rows, dtype=dtype)
        start, stop = 0, buffer_rows
        while numlines > 0:
            buffer_rows = min(numlines, max_buffer_rows)
            # if buffer_rows < len(chunk):
            #    # last chunk is smaller
            #    chunk = np.empty(buffer_rows, dtype=dtype)
            chunk_indices = indices[start:stop]
            # as of PyTables 3.2.2, read_coordinates does not support out=
            # table.read_coordinates(chunk_indices, out=chunk)
            chunk = table.read_coordinates(chunk_indices)
            ca[start:stop] = chunk
            start += buffer_rows
            stop += buffer_rows
            numlines -= buffer_rows
        return ca

    def add_and_drop_fields(self, names_to_keep, output_fields, default_values):
        """modify inplace.
        Only passing output_fields is not enough because one may want to reset a field data (see issue 227).
        """

        output_dtype = np.dtype(output_fields)
        output_names = set(output_dtype.names)
        input_names = set(self.dtype.names)
        # drop extra fields
        for name in input_names - set(names_to_keep):
            del self[name]

        # add missing fields
        length = len(self)
        if default_values is None:
            default_values = {}
        for name in output_names - set(self.dtype.names):
            self[name] = get_default_vector(length, output_dtype[name], default_values[name])


class LColumnArray(object):
    def __init__(self, array=None, axes=None):
        if array is not None:
            columns = {}
            if isinstance(array, (np.ndarray, ColumnArray)):
                if axes is None and 'id' in array.dtype.names:
                    axes = la.AxisCollection([la.Axis(array['id'], 'id')])
                self.axes = axes
                self.dtype = array.dtype
                for name in array.dtype.names:
                    columns[name] = self._prepare_column(array[name].copy())
                self.columns = columns
            elif isinstance(array, list):
                assert isinstance(axes, la.AxisCollection)
                # array is a list of (name, column) pairs
                self.axes = axes
                self.dtype = np.dtype([(name, column.dtype)
                                       for name, column in array])
                for name, column in array:
                    columns[name] = self._prepare_column(column)
                self.columns = columns
            else:
                raise TypeError('invalid array', array)
        else:
            assert axes is None
            self.axes = None
            self.dtype = None
            self.columns = {}

    def __getitem__(self, key):
        if isinstance(key, basestring):
            return self.columns[key]
        else:
            # int, slice, ndarray or bool LArray
            if isinstance(key, la.Array):
                assert key.ndim == 1 and np.issubdtype(key.dtype, np.bool_)
                key = filter_to_indices(key.data)
            assert not isinstance(key, la.Array)
            ca = LColumnArray()
            ca.axes = la.AxisCollection(self.axes.id.subaxis(key))
            ca.columns = {colname: la.Array(colvalue.data[key], ca.axes)
                          for colname, colvalue in self.columns.items()}
            ca.dtype = self.dtype
            return ca

    def __setitem__(self, key, value):
        """does not copy value except if a type conversion is necessary"""

        if isinstance(key, basestring):
            value = self._prepare_column(value)
            if key in self.columns:
                # converting to existing dtype
                if value.dtype != self.dtype[key]:
                    value = value.astype(self.dtype[key])
                self.columns[key] = value
            else:
                # adding a new column so we need to update the dtype
                assert isinstance(value, la.Array)
                self.columns[key] = value
                self._update_dtype()
        else:
            # key is int, slice or ndarray, value is an LColumnArray
            # assert isinstance(value, LColumnArray)
            # assert self.id_axis.subaxis(key).equals(value.id_axis)
            for name, column in self.columns.items():
                assert isinstance(column, la.Array)
                column.data[key] = value[name]

    def _prepare_column(self, value):
        id_axis = self.axes.id
        if isinstance(value, np.ndarray) and value.shape:
            if len(value) != len(id_axis):
                raise ValueError("could not broadcast input array from shape ({}) into shape ({})"
                                 .format(len(value), len(id_axis)))
            return la.Array(value, self.axes)
        elif isinstance(value, la.Array) and value.shape:
            value_axis = value.axes.id
            if not value_axis.iscompatible(id_axis):
                raise ValueError("incompatible id axis between value column ({}) and LColumnArray ({})"
                                 .format(repr(value_axis), repr(id_axis)))
            return value
        else:
            # expand scalars (like ndarray does) so that we don't have to
            # check isinstance(x, ndarray) and x.shape everywhere
            # TODO: create ConstantLArray(length, value, dtype)
            return la.full(self.axes, value, dtype=gettype(value))

    # FIXME: this is broken (but unused)
    def put(self, indices, values, mode='raise'):
        for name, column in self.columns.items():
            assert isinstance(column, la.Array)
            column.put(indices, values[name], mode)

    @property
    def nbytes(self):
        return sum(v.nbytes for v in self.columns.values())

    def __delitem__(self, key):
        del self.columns[key]
        self._update_dtype()

    def _update_dtype(self):
        # handle fields already present (iterate over old dtype to preserve order)
        if self.dtype is not None:
            old_fields = self.dtype.names
            fields = [(name, self.dtype[name])
                      for name in old_fields
                      if name in self.columns]
        else:
            old_fields = []
            fields = []
        # add new fields (not already handled)
        old_fields = set(old_fields)
        fields += [(name, column.dtype)
                   for name, column in self.columns.items()
                   if name not in old_fields]
        self.dtype = np.dtype(fields)

    def __len__(self):
        return len(self.axes.id) if self.axes is not None else 0

    # this is an inplace method so the old memory for each column can be freed before all columns have been changed
    def keep(self, key):
        """key must be a vector of int indices"""

        if not isinstance(key, np.ndarray):
            raise ValueError("key is {} and not ndarray".format(type(key).__name__))
        if not np.issubdtype(key.dtype, np.int_):
            raise ValueError("key dtype is {} and not int-like".format(key.dtype))
        self.axes = la.AxisCollection(self.axes.id.subaxis(key))

        # using gc.collect() after each column update frees a bit of memory but slows things down significantly.
        for name, column in self.columns.items():
            # intentionally not using column[key] to avoid computing the new id axis for each column
            self.columns[name] = la.Array(column.data[key], self.axes)

    # this is an inplace method so the old memory for each column can be freed before all columns have been changed
    def append(self, array):
        assert array.dtype == self.dtype, (array.dtype, self.dtype)
        assert all(isinstance(array[name], la.Array) for name in array.dtype.names)
        self.axes = la.AxisCollection(self.axes.id.extend(array.axes.id))

        # using gc.collect() after each column update frees a bit of memory but slows things down significantly.
        for name, column in self.columns.items():
            # intentionally not using la.concat() to avoid computing the new id axis for each column
            # XXX: we could update the LArray object inplace (only change .data and .axes) but I don't think it's
            #      worth it
            self.columns[name] = la.Array(np.concatenate((column.data, array[name].data)), self.axes)

    def append_to_table(self, table, buffersize=10 * MB):
        ca = ColumnArray([(k, v.data) for k, v in self.columns.items()])
        append_carray_to_table(ca, table, buffersize=buffersize)

    @classmethod
    def empty(cls, axes, dtype):
        ca = cls()
        if isinstance(axes, int):
            if axes == 0:
                axes = np.empty(0, dtype=int)
            axes = la.AxisCollection(la.Axis(axes, 'id'))
        assert isinstance(axes, la.AxisCollection)
        for name in dtype.names:
            ca.columns[name] = la.empty(axes, dtype=dtype[name])
        ca.dtype = dtype
        ca.axes = axes
        return ca

    @classmethod
    def default_array(cls, axes, dtype, default_values):
        ca = cls()
        assert isinstance(axes, la.AxisCollection)
        for name in dtype.names:
            coldtype = dtype[name]
            coldefault = get_default_value(coldtype, default_values.get(name))
            ca.columns[name] = la.full(axes, coldefault, dtype=coldtype)
        ca.dtype = dtype
        ca.axes = axes
        return ca

    @classmethod
    def from_table(cls, table, start=0, stop=None, buffersize=10 * MB):
        # reading a table one column at a time is very slow, this is why this
        # function is even necessary
        if stop is None:
            stop = len(table)
        dtype = table.dtype
        max_buffer_rows = buffersize // dtype.itemsize
        numlines = stop - start
        # start with a wildcard id axis (we will replace it later)
        ca = cls.empty(numlines, dtype)
        buffer_rows = min(numlines, max_buffer_rows)
        chunk = np.empty(buffer_rows, dtype=dtype)
        array_start = 0
        table_start = start
        while numlines > 0:
            buffer_rows = min(numlines, max_buffer_rows)
            if buffer_rows < len(chunk):
                # last chunk is smaller
                chunk = np.empty(buffer_rows, dtype=dtype)
            table.read(table_start, table_start + buffer_rows, out=chunk)
            ca[array_start:array_start + buffer_rows] = chunk
            table_start += buffer_rows
            array_start += buffer_rows
            numlines -= buffer_rows
        # intentionally modifying the id axis inplace because it is referenced in each column (and in the LCA)
        ca.axes.id.labels = ca['id']
        return ca

    def to_carray(self):
        return ColumnArray([(name, column.data) for name, column in self.columns.items()])

    @classmethod
    def from_table_coords(cls, table, indices, buffersize=10 * MB):
        dtype = table.dtype
        max_buffer_rows = buffersize // dtype.itemsize
        numlines = len(indices)
        # start with a wildcard id axis (we will replace it later)
        ca = cls.empty(numlines, dtype)
        buffer_rows = min(numlines, max_buffer_rows)
        # we don't need a buffer because table.read_coordinates allocates it (see below)
        # chunk = np.empty(buffer_rows, dtype=dtype)
        start, stop = 0, buffer_rows
        while numlines > 0:
            buffer_rows = min(numlines, max_buffer_rows)
            # if buffer_rows < len(chunk):
            #    # last chunk is smaller
            #    chunk = np.empty(buffer_rows, dtype=dtype)
            chunk_indices = indices[start:stop]
            # as of PyTables 3.2.2, read_coordinates does not support out=
            # table.read_coordinates(chunk_indices, out=chunk)
            chunk = table.read_coordinates(chunk_indices)
            ca[start:stop] = chunk
            start += buffer_rows
            stop += buffer_rows
            numlines -= buffer_rows
        # intentionally modifying the id_axis inplace because it is referenced in each column (and in the LCA)
        ca.axes.id.labels = ca['id']
        return ca

    def add_and_drop_fields(self, names_to_keep, output_fields, default_values):
        """modify inplace.

        Only passing output_fields is not enough because one may want to reset a field data (see issue 227).
        """

        output_dtype = np.dtype(output_fields)
        output_names = set(output_dtype.names)
        input_names = set(self.dtype.names)
        # drop extra fields
        for name in input_names - set(names_to_keep):
            del self[name]

        # add missing fields
        length = len(self)
        if default_values is None:
            default_values = {}
        for name in output_names - set(self.dtype.names):
            self[name] = get_default_value(output_dtype[name], default_values[name])


def get_fields(array):
    dtype = array.dtype
    return [(name, normalize_type(dtype[name].type)) for name in dtype.names]


def assert_valid_type(array, wanted_type, context=None):
    if isinstance(wanted_type, list):
        wanted_fields = wanted_type
        # extract types from field description and normalise to python types
        actual_fields = get_fields(array)

        # check that all required fields are present
        wanted_names = set(name for name, _ in wanted_fields)
        actual_names = set(name for name, _ in actual_fields)
        missing = wanted_names - actual_names
        if missing:
            raise Exception("Missing field(s) in hdf5 input file: %s"
                            % ', '.join(missing))

        # check that types match
        common_fields1 = sorted((name, type_) for name, type_ in actual_fields
                                if name in wanted_names)
        common_fields2 = sorted((name, type_) for name, type_ in wanted_fields
                                if name in actual_names)
        bad_fields = []
        for (name1, t1), (name2, t2) in zip(common_fields1, common_fields2):
            # this can happen if we have duplicates in wanted_fields
            assert name1 == name2, "%s != %s" % (name1, name2)
            if t1 != t2:
                bad_fields.append((name1, t2.__name__, t1.__name__))
        if bad_fields:
            bad_fields_str = "\n".join(" - %s: %s instead of %s" % f
                                       for f in bad_fields)
            raise Exception("Field types in hdf5 input file differ from those "
                            "defined in the simulation:\n%s" % bad_fields_str)
    else:
        assert isinstance(wanted_type, type)
        actual_type = normalize_type(array.dtype.type)
        if actual_type != wanted_type:
            raise Exception("Field type for '%s' in hdf5 input file is '%s' "
                            "instead of '%s'" % (context, actual_type.__name__,
                                                 wanted_type.__name__))


def add_and_drop_fields(array, output_fields, default_values=None,
                        output_array=None):
    default_values = default_values if default_values is not None else dict()
    output_dtype = np.dtype(output_fields)
    output_names = set(output_dtype.names)
    input_names = set(array.dtype.names)

    common_fields = output_names & input_names
    missing_fields = output_names - input_names
    if output_array is None:
        output_array = np.empty(len(array), dtype=output_dtype)
        for fname in missing_fields:
            output_array[fname] = get_default_value(output_array[fname],
                                                    default_values.get(fname))
    else:
        assert output_array.dtype == output_dtype
    for fname in common_fields:
        output_array[fname] = array[fname]
    return output_array


def merge_subset_in_array(output, id_to_rownum, subset, first=False,
                          default_values=None):
    default_values = default_values if default_values is not None else {}
    if subset.dtype == output.dtype and len(subset) == len(output):
        return subset
    elif subset.dtype == output.dtype:
        safe_put(output, id_to_rownum[subset['id']], subset)
        return output

    output_names = output.dtype.names
    subset_names = subset.dtype.names
    names_to_copy = set(subset_names) & set(output_names)
    if len(subset) == len(output):
        for fname in names_to_copy:
            output[fname] = subset[fname]
        return output
    else:
        rownums = id_to_rownum[subset['id']]
        # TODO: this is a gross approximation, more research is needed to get
        # a better threshold. It might also depend on "first".
        if len(names_to_copy) > len(output_names) / 2:
            if first:
                subset_all_cols = np.empty(len(subset), dtype=output.dtype)
                for fname in set(output_names) - set(subset_names):
                    default_value = default_values.get(fname, None)
                    subset_all_cols[fname] = \
                        get_default_value(subset_all_cols[fname], default_value)
            else:
                subset_all_cols = output[rownums]
                # Note that all rows which correspond to rownums == -1 have
                # wrong values (they have the value of the last row) but it is
                # not necessary to correct them since they will not be copied
                # back into output_array.
                # np.putmask(subset_all_cols, rownums == -1, missing_row)
            for fname in names_to_copy:
                subset_all_cols[fname] = subset[fname]
            safe_put(output, rownums, subset_all_cols)
        else:
            for fname in names_to_copy:
                safe_put(output[fname], rownums, subset[fname])
        return output


def merge_array_records(array1, array2):
    """
    array1 & array2
    data in array2 overrides data in array1
    """
    assert len(array1) == len(array2) == 1
    fields1 = get_fields(array1)
    fields2 = get_fields(array2)

    names1 = set(array1.dtype.names)
    names2 = set(array2.dtype.names)
    fields_notin1 = [(name, type_) for name, type_ in fields2
                     if name not in names1]
    output_fields = fields1 + fields_notin1
    output = np.empty(1, np.dtype(output_fields))
    for fname in names1 - names2:
        output[fname] = array1[fname]
    for fname in names2:
        output[fname] = array2[fname]
    return output


def merge_arrays(array1, array2, result_fields='union', default_values=None):
    """
    data in array2 overrides data in array1
    both arrays must have 'id' fields
    """

    fields1 = get_fields(array1)
    fields2 = get_fields(array2)

    # TODO: check that common fields have the same type
    if result_fields == 'union':
        names1 = set(array1.dtype.names)
        fields_notin1 = [(name, type_) for name, type_ in fields2
                         if name not in names1]
        output_fields = fields1 + fields_notin1
    elif result_fields == 'array1':
        output_fields = fields1
    else:
        raise ValueError('%s in not a valid value for result_fields argument' %
                         result_fields)

    output_dtype = np.dtype(output_fields)

    ids1 = array1['id']
    ids2 = array2['id']
    all_ids = np.union1d(ids1, ids2)
    max_id = all_ids[-1]

    # compute new id_to_rownum
    id_to_rownum = np.full(max_id + 1, -1, dtype=int)
    for rownum, rowid in enumerate(all_ids):
        id_to_rownum[rowid] = rownum

    # 1) create resulting array
    ids1_complete = len(ids1) == len(all_ids)
    ids2_complete = len(ids2) == len(all_ids)
    output_is_arr1 = array1.dtype == output_dtype and ids1_complete
    output_is_arr2 = array2.dtype == output_dtype and ids2_complete
    arr1_complete = set(fields1) >= set(output_fields) and ids1_complete
    arr2_complete = set(fields2) >= set(output_fields) and ids2_complete
    if output_is_arr2:
        output_array = array2
    elif output_is_arr1:
        # TODO: modifying array1 in-place suits our particular needs for now
        # but it should really be a (non-default) option
        output_array = array1
    elif arr1_complete or arr2_complete:
        output_array = np.empty(len(all_ids), dtype=output_dtype)
    else:
        output_array = get_default_array(len(all_ids), output_dtype,
                                         default_values)

    # 2) copy data from array1 (if it will not be overridden)
    if not arr2_complete:
        output_array = merge_subset_in_array(output_array, id_to_rownum,
                                             array1, first=True,
                                             default_values=default_values)

    # 3) copy data from array2
    if not output_is_arr2:
        output_array = merge_subset_in_array(output_array, id_to_rownum, array2,
                                             default_values=default_values)

    return output_array, id_to_rownum


def append_table(input_table, output_table, chunksize=10000, condition=None,
                 stop=None, show_progress=False, default_values=None):

    if input_table.dtype != output_table.dtype:
        output_fields = get_fields(output_table)
    else:
        output_fields = None

    if stop is None:
        numrows = len(input_table)
    else:
        numrows = stop

    if not chunksize:
        chunksize = numrows

    num_chunks, remainder = divmod(numrows, chunksize)
    if remainder > 0:
        num_chunks += 1

    if output_fields is not None:
        expanded_data = get_default_array(chunksize, np.dtype(output_fields),
                                          default_values)

    # noinspection PyUnusedLocal
    def copy_chunk(chunk_idx, chunk_num):
        chunk_start = chunk_num * chunksize
        chunk_stop = min(chunk_start + chunksize, numrows)
        if condition is not None:
            input_data = input_table.readWhere(condition, start=chunk_start,
                                               stop=chunk_stop)
        else:
            input_data = input_table.read(chunk_start, chunk_stop)

        if output_fields is not None:
            # use our pre-allocated buffer (except for the last chunk)
            if len(input_data) == len(expanded_data):
                output_data = add_and_drop_fields(input_data, output_fields,
                                                  default_values, expanded_data)
            else:
                output_data = add_and_drop_fields(input_data, output_fields,
                                                  default_values)
        else:
            output_data = input_data

        output_table.append(output_data)
        output_table.flush()

    if show_progress:
        loop_wh_progress(copy_chunk, range(num_chunks))
    else:
        for chunk in range(num_chunks):
            copy_chunk(chunk, chunk)

    return output_table


# noinspection PyProtectedMember
def copy_table(input_table, output_node, output_dtype=None,
               chunksize=10000, condition=None, stop=None, show_progress=False,
               default_values=None, **kwargs):
    complete_kwargs = {'title': input_table._v_title}
#                       'filters': input_table.filters}
    output_file = output_node._v_file
    complete_kwargs.update(kwargs)
    if output_dtype is None:
        output_dtype = input_table.dtype
    output_table = output_file.create_table(output_node, input_table.name,
                                            output_dtype, **complete_kwargs)
    return append_table(input_table, output_table, chunksize, condition,
                        stop=stop, show_progress=show_progress,
                        default_values=default_values)


# XXX: should I make a generic n-way array merge out of this?
# this is a special case though because:
# 1) all arrays have the same columns
# 2) we have id_to_rownum already computed for each array
def build_period_array(input_table, fields_to_keep, output_fields, input_rows,
                       input_index, start_period, default_values):
    periods_before = [p for p in input_rows.keys() if p <= start_period]
    if not periods_before:
        id_to_rownum = np.empty(0, dtype=int)
        output_array = LColumnArray.empty(0, np.dtype(output_fields))
        return output_array, id_to_rownum

    periods_before.sort()
    # take the last period which we have data for
    target_period = periods_before[-1]

    # computing is_present
    max_id = len(input_index[target_period]) - 1
    period_id_to_rownum = None
    present_in_period = None
    is_present = np.zeros(max_id + 1, dtype=bool)
    for period in periods_before:
        period_id_to_rownum = input_index[period]
        present_in_period = period_id_to_rownum != -1
        present_in_period.resize(max_id + 1, refcheck=False)
        is_present |= present_in_period

    # if all individuals are present in the target period, we are done already!
    if np.array_equal(present_in_period, is_present):
        start, stop = input_rows[target_period]
        input_array = LColumnArray.from_table(input_table, start, stop)
        input_array.add_and_drop_fields(fields_to_keep, output_fields, default_values)
        return input_array, period_id_to_rownum

    # building id_to_rownum for the target period
    id_to_rownum = np.full(max_id + 1, -1, dtype=int)
    rownum = 0
    for row_id, present in enumerate(is_present):
        if present:
            id_to_rownum[row_id] = rownum
            rownum += 1

    # computing the source row for each destination row
    # we loop over the periods before start_period in reverse order
    output_array_source_rows = np.full(rownum, -1, dtype=int)
    for period in periods_before[::-1]:
        start, stop = input_rows[period]
        input_rownums = np.arange(start, stop)

        input_id_to_rownum = input_index[period]
        id_is_in_period = input_id_to_rownum != -1

        # which output rows are filled by input for this period
        output_rownums = id_to_rownum[np.where(id_is_in_period)]

        # get source rows (in the global array) for individuals in this period
        source_rows = output_array_source_rows[output_rownums]

        # if their source row is already known, leave them alone
        need_update = source_rows == -1

        # global indices of rows which are not set yet (for this period)
        rows_to_update = output_rownums[need_update]

        # source row for those rows
        local_source_rows = input_rownums[need_update]

        # update the source row for those rows
        safe_put(output_array_source_rows, rows_to_update, local_source_rows)

        if np.all(output_array_source_rows != -1):
            break

    # reading data
    output_array = LColumnArray.from_table_coords(input_table, output_array_source_rows)
    output_array.add_and_drop_fields(fields_to_keep, output_fields, default_values)
    return output_array, id_to_rownum


def index_table(table):
    """
    table is an iterable of rows, each row is a mapping (name -> value).
    Rows must contain at least 'period' and 'id' columns and must be sorted
    by period.
    """
    rows_per_period = {}
    id_to_rownum_per_period = {}
    temp_id_to_rownum = []
    max_id_so_far = -1
    current_period = None
    start_row = None
    for idx, row in enumerate(table):
        period, row_id = row['period'], row['id']
        if period != current_period:
            if current_period is not None and period < current_period:
                msg = "data is not ordered by period " \
                      "({} at data line {} is < {})"
                raise Exception(msg.format(period, idx + 1, current_period))
            if start_row is not None:
                rows_per_period[current_period] = start_row, idx
                # assumes the data is sorted on period then id
                id_to_rownum = np.array(temp_id_to_rownum)
                id_to_rownum_per_period[current_period] = id_to_rownum
                temp_id_to_rownum = [-1] * (max_id_so_far + 1)
            start_row = idx
            current_period = period
        if row_id > max_id_so_far:
            extra = [-1] * (row_id - max_id_so_far)
            temp_id_to_rownum.extend(extra)
        if temp_id_to_rownum[row_id] != -1:
            msg = "duplicate row for id {} for period {} (at data line {})"
            # idx + 1 is correct for ViTables, which starts counting at 1, but
            # is still off by one (or more) for .csv files because of headers
            # and comments
            raise Exception(msg.format(row_id, period, idx + 1))
        temp_id_to_rownum[row_id] = idx - start_row
        max_id_so_far = max(max_id_so_far, row_id)
    if current_period is not None:
        rows_per_period[current_period] = (start_row, len(table))
        id_to_rownum_per_period[current_period] = np.array(temp_id_to_rownum)
    return rows_per_period, id_to_rownum_per_period


def index_table_light(table, index='period'):
    """
    table is an iterable of rows, each row is a mapping (name -> value)
    Rows must contain the index column and must be sorted by that column.
    Returns a dict: {index_value: start_row, stop_row}
    """
    rows_per_period = {}
    current_value = None
    start_row = None
    # I don't know why but my attempts to only retrieve one column
    # made the function slower, not faster (this is only used in diff_h5 &
    # merge_h5 though).
    for idx, row in enumerate(table):
        value = row[index]
        if value != current_value:
            # 0 > None is True
            if value < current_value:
                msg = "data is not ordered by {} ({} at data line {} is < {})"
                raise Exception(msg.format(index, value, idx + 1, current_value))
            if start_row is not None:
                rows_per_period[current_value] = (start_row, idx)
            start_row = idx
            current_value = value
    if current_value is not None:
        rows_per_period[current_value] = (start_row, len(table))
    return rows_per_period


class IndexedTable(object):
    def __init__(self, table, period_index, id2rownum_per_period):
        self.table = table
        self.period_index = period_index
        self.id2rownum_per_period = id2rownum_per_period

    # In the future, we will probably want a more flexible interface, but
    # it will need a lot more machinery (query language and so on) so let's
    # stick with something simple for now.
    def read(self, period, ids=None, field=None):
        if period not in self.period_index:
            raise Exception('no data for period %d' % period)
        if ids is None:
            start, stop = self.period_index[period]
            return self.table.read(start=start, stop=stop, field=field)
        else:
            raise NotImplementedError('reading only some ids is not '
                                      'implemented yet')

    # XXX: use __contains__?
    def has_period(self, period):
        return period in self.period_index

    @property
    def base_period(self):
        return min(self.period_index.keys())


class DataSet(object):
    pass


def load_path_globals(globals_def):
    localdir = config.input_directory
    globals_data = {}
    for name, global_def in globals_def.items():
        # skip constants
        if not isinstance(global_def, dict):
            continue
        # skip globals stored in the .h5 input file
        if 'path' not in global_def:
            continue

        kind, info = load_def(localdir, name, global_def, [])
        if kind == 'table':
            fields, numlines, datastream, csvfile = info
            array = stream_to_array(fields, datastream, numlines)
            csvfile.close()
        else:
            assert kind == 'ndarray'
            array = info
        globals_data[name] = array
    return globals_data


def handle_constant_globals(globals_def):
    globals_data = {k: gdef for k, gdef in globals_def.items() if np.isscalar(gdef)}
    return globals_data


def index_tables(globals_def, entities, fpath):
    print("reading data from %s ..." % fpath)
    input_file = tables.open_file(fpath)
    try:
        input_root = input_file.root

        def must_load_from_input_file(gdef):
            return isinstance(gdef, dict) and 'path' not in gdef
        any_global_from_input_file = any(must_load_from_input_file(gdef) for gdef in globals_def.values())
        if any_global_from_input_file and 'globals' not in input_root:
            raise Exception('could not find any globals in the input data file '
                            '(but some are declared in the simulation file)')

        globals_data = load_path_globals(globals_def)
        constant_globals_data = handle_constant_globals(globals_def)
        globals_data.update(constant_globals_data)
        globals_node = getattr(input_root, 'globals', None)
        for name, global_def in globals_def.items():
            # already loaded from another source (path)
            if name in globals_data:
                continue

            if name not in globals_node:
                raise Exception("could not find 'globals/%s' in the input "
                                "data file" % name)

            global_data = getattr(globals_node, name)

            global_type = global_def.get('type', global_def.get('fields'))
            # TODO: move the checking (assertValidType) to a separate function
            assert_valid_type(global_data, global_type, context=name)
            array = global_data.read()
            if isinstance(global_type, list):
                # make sure we do not keep in memory columns which are
                # present in the input file but where not asked for by the
                # modeller. They are not accessible anyway.
                array = add_and_drop_fields(array, global_type)
            attrs = global_data.attrs
            dim_names = getattr(attrs, 'dimensions', None)
            if dim_names is not None:
                # we serialise dim_names as a numpy array so that it is
                # stored as a native hdf type and not a pickle but we
                # prefer to work with simple lists
                # also files serialized using Python2 are "bytes" not "str"
                dim_names = [str(dim_name) for dim_name in dim_names]
                pvalues = [getattr(attrs, 'dim%d_pvalues' % i)
                           for i in range(len(dim_names))]
                axes = [la.Axis(labels, axis_name)
                        for axis_name, labels in zip(dim_names, pvalues)]
                array = la.Array(array, axes)
            globals_data[name] = array

        input_entities = input_root.entities

        entities_tables = {}
        print(" * indexing tables")
        for ent_name, entity in entities.items():
            print("    -", ent_name, "...", end=' ')

            table = getattr(input_entities, ent_name)
            assert_valid_type(table, list(entity.fields.in_input.name_types))

            rows_per_period, id_to_rownum_per_period = \
                timed(index_table, table)
            indexed_table = IndexedTable(table, rows_per_period,
                                         id_to_rownum_per_period)
            entities_tables[ent_name] = indexed_table
    except:
        input_file.close()
        raise

    return input_file, {'globals': globals_data, 'entities': entities_tables}


class DataSource(object):
    def close(self):
        pass


class DataSink(object):
    def close(self):
        pass


class VoidSource(DataSource):
    def load(self, globals_def, entities):
        return {'globals': load_path_globals(globals_def), 'entities': {}}


class H5Source(DataSource):
    def __init__(self, input_path):
        self.h5in = None
        self.input_path = input_path

    def load(self, globals_def, entities):
        h5file, dataset = index_tables(globals_def, entities, self.input_path)
        entities_tables = dataset['entities']
        for ent_name, entity in entities.items():
            table = entities_tables[ent_name]
            # entity.indexed_input_table = table
            entity.input_index = table.id2rownum_per_period
            entity.input_rows = table.period_index
            entity.input_table = table.table
            entity.base_period = table.base_period
            entity.id_axis_per_period = index_per_period_to_axis_per_period(table.id2rownum_per_period)
        self.h5in = h5file
        return dataset

    def close(self):
        if self.h5in is not None:
            self.h5in.close()

    def as_fake_output(self, dataset, entities):
        entities_tables = dataset['entities']
        for ent_name, entity in entities.items():
            table = entities_tables[ent_name]
            entity.output_index = table.id2rownum_per_period
            entity.output_rows = table.period_index
            entity.table = table.table


class H5Sink(DataSink):
    def __init__(self, output_path):
        self.output_path = output_path
        self.h5out = None

    def prepare(self, globals_def, entities, input_dataset, start_period):
        """copy input (if any) to output and create output index"""
        output_file = tables.open_file(self.output_path, mode="w")

        try:
            globals_data = input_dataset.get('globals')
            if globals_data is not None:
                output_globals = output_file.create_group("/", "globals",
                                                          "Globals")
                for k, g_def in globals_def.items():
                    # Do not save global constants nor globals loaded from external (.csv) files.
                    if np.isscalar(g_def) or 'path' in g_def:
                        continue

                    anyarray_to_disk(output_globals, k, globals_data[k])

            entities_tables = input_dataset['entities']
            output_entities = output_file.create_group("/", "entities",
                                                       "Entities")
            output_file.create_group("/", "indexes", "Indexes")
            output_file.create_group("/", "axes", "Axes")

            print(" * copying tables")
            for ent_name, entity in entities.items():
                print("    -", ent_name, "...", end=' ')
                entity.output_index_node = output_file.create_group("/indexes", ent_name)
                entity.output_axes_node = output_file.create_group("/axes", ent_name)
                if not entity.fields.in_output:
                    print("skipped (no column in output)")
                    continue

                start_time = time.time()

                # main table
                table = entities_tables.get(ent_name)
                if table is not None:
                    input_rows = table.period_index
                    output_rows = dict((p, rows)
                                       for p, rows in input_rows.items()
                                       if p < start_period)
                    if output_rows:
                        # stoprow = last row of the last period before
                        #           start_period
                        _, stoprow = input_rows[max(output_rows.keys())]
                    else:
                        stoprow = 0

                    output_table = copy_table(table.table, output_entities,
                                              entity.fields.in_output.dtype,
                                              stop=stoprow,
                                              show_progress=True,
                                              default_values=entity.fields.default_values)
                    output_index = table.id2rownum_per_period.copy()
                else:
                    output_rows = {}
                    output_table = output_file.create_table(
                        output_entities, entity.name,
                        entity.fields.in_output.dtype,
                        title="%s table" % entity.name)
                    output_index = {}

                # entity.indexed_output_table = IndexedTable(output_table,
                #                                            output_rows,
                #                                            output_index)
                entity.output_index = output_index
                entity.output_rows = output_rows
                entity.table = output_table
                print("done (%s elapsed)." % time2str(time.time() - start_time))
        except:
            output_file.close()
            raise
        self.h5out = output_file

    def close(self):
        if self.h5out is not None:
            self.h5out.close()


def entities_from_h5(fpath):
    from liam2.entities import Entity
    h5in = tables.open_file(fpath)
    h5root = h5in.root
    entities = {}
    for table in h5root.entities:
        entity = Entity.from_table(table)
        entities[entity.name] = entity
    globals_def = {}
    if hasattr(h5root, 'globals'):
        for table in h5root.globals:
            if isinstance(table, tables.Array):
                global_def = {'type': normalize_type(table.dtype.type)}
            else:
                global_def = {'fields': get_fields(table)}
            globals_def[table.name] = global_def
    h5in.close()
    return globals_def, entities


def id_to_rownum_to_id_axis(id_to_rownum):
    return la.Axis([id_ for id_, rownum in enumerate(id_to_rownum) if rownum != -1], 'id')


def index_per_period_to_axis_per_period(index_per_period):
    return {period: id_to_rownum_to_id_axis(id_to_rownum)
            for period, id_to_rownum in index_per_period.items()}
