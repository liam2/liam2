# encoding: utf-8
from __future__ import print_function

import time

import tables
import numpy as np
import config

from expr import (normalize_type, get_missing_value, get_missing_record,
                  get_missing_vector, gettype)
from utils import loop_wh_progress, time2str, safe_put, LabeledArray
from importer import load_def, stream_to_array

MB = 2 ** 20


def append_carray_to_table(array, table, numlines=None, buffersize=10 * MB):
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
            chunk[fieldname] = array[fieldname][start:stop]
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
        columns = {}
        if array is not None:
            if isinstance(array, (np.ndarray, ColumnArray)):
                for name in array.dtype.names:
                    columns[name] = array[name].copy()
                self.dtype = array.dtype
                self.columns = columns
            elif isinstance(array, list):
                for name, column in array:
                    columns[name] = column
                self.dtype = np.dtype([(name, column.dtype)
                                       for name, column in array])
                self.columns = columns
            else:
                # TODO: make a property instead?
                self.dtype = None
                self.columns = columns
        else:
            self.dtype = None
            self.columns = columns

    def __getitem__(self, key):
        if isinstance(key, basestring):
            return self.columns[key]
        else:
            # int, slice, ndarray
            ca = ColumnArray()
            for name, colvalue in self.columns.iteritems():
                ca[name] = colvalue[key]
            ca.dtype = self.dtype
            return ca

    def __setitem__(self, key, value):
        """does not copy value except if a type conversion is necessary"""

        if isinstance(key, basestring):
            if isinstance(value, np.ndarray) and value.shape:
                column = value
            else:
                # expand scalars (like ndarray does) so that we don't have to
                # check isinstance(x, ndarray) and x.shape everywhere
                column = np.empty(len(self), dtype=gettype(value))
                column.fill(value)

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
#            for k, v in self.columns.iteritems():
#                ids.setdefault(id(v), set()).add(k)
#            dupes = [v for k, v in ids.iteritems() if len(v) > 1]
#            if dupes:
#                print "aliases", dupes
        else:
            # int, slice, ndarray
            for name, column in self.columns.iteritems():
                column[key] = value[name]

    def put(self, indices, values, mode='raise'):
        for name, column in self.columns.iteritems():
            column.put(indices, values[name], mode)

    @property
    def nbytes(self):
        return sum(v.nbytes for v in self.columns.itervalues())

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
                   for name, column in self.columns.iteritems()
                   if name not in old_fields]
        self.dtype = np.dtype(fields)

    def __len__(self):
        if len(self.columns):
            anycol = self.columns.itervalues().next()
            return len(anycol)
        else:
            return 0

    def keep(self, key):
        """key can be either a vector of int indices or boolean filter"""

        # using gc.collect() after each column update frees a bit of memory
        # but slows things down significantly.
        for name, column in self.columns.iteritems():
            self.columns[name] = column[key]

    def append(self, array):
        assert array.dtype == self.dtype, (array.dtype, self.dtype)
        # using gc.collect() after each column update frees a bit of memory
        # but slows things down significantly.
        for name, column in self.columns.iteritems():
            self.columns[name] = np.concatenate((column, array[name]))

    def append_to_table(self, table, buffersize=10 * 2 ** 20):
        append_carray_to_table(self, table, buffersize=buffersize)

    @classmethod
    def empty(cls, length, dtype):
        ca = cls()
        for name in dtype.names:
            ca.columns[name] = np.empty(length, dtype[name])
        ca.dtype = dtype
        return ca

    @classmethod
    def from_table(cls, table, start=0, stop=None, buffersize=10 * 2 ** 20):
        # reading a table one column at a time is very slow, this is why this
        # function is even necessary
        if stop is None:
            stop = len(table)
        dtype = table.dtype
        max_buffer_rows = buffersize // dtype.itemsize
        numlines = stop - start
        ca = cls.empty(numlines, dtype)
#        buffer_rows = min(numlines, max_buffer_rows)
#        chunk = np.empty(buffer_rows, dtype=dtype)
        array_start = 0
        table_start = start
        while numlines > 0:
            buffer_rows = min(numlines, max_buffer_rows)
#            if buffer_rows < len(chunk):
                # last chunk is smaller
#                chunk = np.empty(buffer_rows, dtype=dtype)
# needs pytables3
#            table.read(table_start, table_start + buffer_rows, out=chunk)
            chunk = table.read(table_start, table_start + buffer_rows)
            ca[array_start:array_start + buffer_rows] = chunk
            table_start += buffer_rows
            array_start += buffer_rows
            numlines -= buffer_rows
        return ca

    @classmethod
    def from_table_coords(cls, table, indices, buffersize=10 * 2 ** 20):
        dtype = table.dtype
        max_buffer_rows = buffersize // dtype.itemsize
        numlines = len(indices)
        ca = cls.empty(numlines, dtype)
        buffer_rows = min(numlines, max_buffer_rows)
#        chunk = np.empty(buffer_rows, dtype=dtype)
        start, stop = 0, buffer_rows
        while numlines > 0:
            buffer_rows = min(numlines, max_buffer_rows)
#            if buffer_rows < len(chunk):
                # last chunk is smaller
#                chunk = np.empty(buffer_rows, dtype=dtype)
            chunk_indices = indices[start:stop]
# needs pytables3
#            table.readCoordinates(chunk_indices, out=chunk)
            chunk = table.readCoordinates(chunk_indices)
            ca[start:stop] = chunk
            start += buffer_rows
            stop += buffer_rows
            numlines -= buffer_rows
        return ca

    def add_and_drop_fields(self, output_fields):
        """modify inplace"""

        output_dtype = np.dtype(output_fields)
        output_names = set(output_dtype.names)
        input_names = set(self.dtype.names)
        
        length = len(self)
        # add missing fields
        for name in output_names - input_names:
            self[name] = get_missing_vector(length, output_dtype[name])
        # delete extra fields
        for name in input_names - output_names:
            del self[name]


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


def add_and_drop_fields(array, output_fields, output_array=None):
    output_dtype = np.dtype(output_fields)
    output_names = set(output_dtype.names)
    input_names = set(array.dtype.names)
    
    common_fields = output_names & input_names
    missing_fields = output_names - input_names
    if output_array is None:
        output_array = np.empty(len(array), dtype=output_dtype)
        for fname in missing_fields:
            output_array[fname] = get_missing_value(output_array[fname])
    else:
        assert output_array.dtype == output_dtype
    for fname in common_fields:
        output_array[fname] = array[fname]
    return output_array


def merge_subset_in_array(output, id_to_rownum, subset, first=False):
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
                    subset_all_cols[fname] = \
                        get_missing_value(subset_all_cols[fname])
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


def merge_arrays(array1, array2, result_fields='union'):
    """data in array2 overrides data in array1"""

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
    id_to_rownum = np.empty(max_id + 1, dtype=int)
    id_to_rownum.fill(-1)
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
        output_array = np.empty(len(all_ids), dtype=output_dtype)
        output_array[:] = get_missing_record(output_array)

    # 2) copy data from array1 (if it will not be overridden)
    if not arr2_complete:
        output_array = merge_subset_in_array(output_array, id_to_rownum,
                                             array1, first=True)

    # 3) copy data from array2
    if not output_is_arr2:
        output_array = merge_subset_in_array(output_array, id_to_rownum, array2)

    return output_array, id_to_rownum


def append_table(input_table, output_table, chunksize=10000, condition=None,
                 stop=None, show_progress=False):
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
        expanded_data = np.empty(chunksize, dtype=np.dtype(output_fields))
        expanded_data[:] = get_missing_record(expanded_data)

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
                                                  expanded_data)
            else:
                output_data = add_and_drop_fields(input_data, output_fields)
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
               **kwargs):
    complete_kwargs = {'title': input_table._v_title}
#                       'filters': input_table.filters}
    output_file = output_node._v_file
    complete_kwargs.update(kwargs)
    if output_dtype is None:
        output_dtype = input_table.dtype
    output_table = output_file.create_table(output_node, input_table.name,
                                            output_dtype, **complete_kwargs)
    return append_table(input_table, output_table, chunksize, condition,
                        stop=stop, show_progress=show_progress)


# XXX: should I make a generic n-way array merge out of this?
# this is a special case though because:
# 1) all arrays have the same columns
# 2) we have id_to_rownum already computed for each array
def build_period_array(input_table, output_fields, input_rows, input_index,
                       start_period):
    periods_before = [p for p in input_rows.iterkeys() if p <= start_period]
    if not periods_before:
        id_to_rownum = np.empty(0, dtype=int)
        output_array = ColumnArray.empty(0, np.dtype(output_fields))
        return output_array, id_to_rownum

    periods_before.sort()
    # take the last period which we have data for
    target_period = periods_before[-1]

    # computing is present
    max_id = len(input_index[target_period]) - 1
    period_id_to_rownum = None
    present_in_period = None
    is_present = np.zeros(max_id + 1, dtype=bool)
    for period in periods_before:
        period_id_to_rownum = input_index[period]
        present_in_period = period_id_to_rownum != -1
        present_in_period.resize(max_id + 1)
        is_present |= present_in_period

    # if all individuals are present in the target period, we are done already!
    if np.array_equal(present_in_period, is_present):
        start, stop = input_rows[target_period]
        input_array = ColumnArray.from_table(input_table, start, stop)
        input_array.add_and_drop_fields(output_fields)
        return input_array, period_id_to_rownum

    # building id_to_rownum for the target period
    id_to_rownum = np.empty(max_id + 1, dtype=int)
    id_to_rownum.fill(-1)
    rownum = 0
    for row_id, present in enumerate(is_present):
        if present:
            id_to_rownum[row_id] = rownum
            rownum += 1

    # computing the source row for each destination row
    # we loop over the periods before start_period in reverse order
    output_array_source_rows = np.empty(rownum, dtype=int)
    output_array_source_rows.fill(-1)
    for period in periods_before[::-1]:
        start, stop = input_rows[period]
        input_rownums = np.arange(start, stop)

        input_id_to_rownum = input_index[period]
        id_is_in_period = input_id_to_rownum != -1

        # which output rows are filled by input for this period
        output_rownums = id_to_rownum[id_is_in_period]

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
    output_array = ColumnArray.from_table_coords(input_table,
                                                 output_array_source_rows)
    output_array.add_and_drop_fields(output_fields)
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
            # 0 > None is True
            if period < current_period:
                raise Exception("data is not time-ordered")
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
    # I don't know whether or not but my attempts to only retrieve one column
    # made the function slower, not faster (this is only used in diff_h5 &
    # merge_h5 though).
    for idx, row in enumerate(table):
        value = row[index]
        if value != current_value:
            # 0 > None is True
            if value < current_value:
                raise Exception("data is not time-ordered")
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

    @property
    def base_period(self):
        return min(self.period_index.keys())


class DataSet(object):
    pass


def load_path_globals(globals_def):
    localdir = config.input_directory
    globals_data = {}
    for name, global_def in globals_def.iteritems():
        if 'path' not in global_def:
            continue
        kind, info = load_def(localdir, name, global_def, [])
        if kind == 'table':
            fields, numlines, datastream, csvfile = info
            array = stream_to_array(fields, datastream, numlines)
        else:
            assert kind == 'ndarray'
            array = info
        globals_data[name] = array
    return globals_data


def index_tables(globals_def, entities, fpath):
    print("reading data from %s ..." % fpath)

    input_file = tables.open_file(fpath)
    try:
        input_root = input_file.root

        if any('path' not in g_def for g_def in globals_def.itervalues()) and \
                'globals' not in input_root:
            raise Exception('could not find any globals in the input data file '
                            '(but some are declared in the simulation file)')

        globals_data = load_path_globals(globals_def)

        globals_node = getattr(input_root, 'globals', None)
        for name, global_def in globals_def.iteritems():
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
                dim_names = list(dim_names)
                pvalues = [getattr(attrs, 'dim%d_pvalues' % i)
                           for i in range(len(dim_names))]
                array = LabeledArray(array, dim_names, pvalues)
            globals_data[name] = array

        input_entities = input_root.entities

        entities_tables = {}
        print(" * indexing tables")
        for ent_name, entity in entities.iteritems():
            print("    -", ent_name, "...", end=' ')

            table = getattr(input_entities, ent_name)
            assert_valid_type(table, list(entity.fields.in_input.name_types))

            start_time = time.time()
            rows_per_period, id_to_rownum_per_period = index_table(table)
            indexed_table = IndexedTable(table, rows_per_period,
                                         id_to_rownum_per_period)
            entities_tables[ent_name] = indexed_table
            print("done (%s elapsed)." % time2str(time.time() - start_time))
    except:
        input_file.close()
        raise

    return input_file, {'globals': globals_data, 'entities': entities_tables}


class DataSource(object):
    pass


# A data source is not necessarily read-only, but should be connected to
# only one file, so in our case we should have one instance for input and the
# other (used both for read and write) for the output.
class H5Data(DataSource):
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def load(self, globals_def, entities):
        h5file, dataset = index_tables(globals_def, entities, self.output_path)
        entities_tables = dataset['entities']
        for ent_name, entity in entities.iteritems():
# this is what should happen
#            entity.indexed_input_table = entities_tables[ent_name]
#            entity.indexed_output_table = entities_tables[ent_name]
            table = entities_tables[ent_name]

            entity.input_index = table.id2rownum_per_period
            entity.input_rows = table.period_index
            entity.input_table = table.table

            entity.output_index = table.id2rownum_per_period
            entity.output_rows = table.period_index
            entity.table = table.table

            entity.base_period = table.base_period

        return h5file, None, dataset['globals']

    def run(self, globals_def, entities, start_period):
        input_file, dataset = index_tables(globals_def, entities,
                                           self.input_path)
        output_file = tables.open_file(self.output_path, mode="w")

        try:
            globals_node = getattr(input_file.root, 'globals', None)
            if globals_node is not None:
                output_globals = output_file.create_group("/", "globals",
                                                          "Globals")
                # index_tables already checks whether all tables exist and
                # are coherent with globals_def
                for name in globals_def:
                    # FIXME: if a globals is both in the input h5 and declared
                    # to be coming from a csv file, it is copied from the h5
                    # file, which is wrong/misleading because it is not used
                    # in the simulation.
                    if name in globals_node:
                        # noinspection PyProtectedMember
                        # FIXME: only copy declared fields
                        getattr(globals_node, name)._f_copy(output_globals)

            entities_tables = dataset['entities']
            output_entities = output_file.create_group("/", "entities",
                                                       "Entities")
            output_indexes = output_file.create_group("/", "indexes", "Indexes")
            print(" * copying tables")
            for ent_name, entity in entities.iteritems():
                print(ent_name, "...")

                # main table

                table = entities_tables[ent_name]

                index_node = output_file.create_group("/indexes", ent_name)
                entity.output_index_node = index_node
                entity.input_index = table.id2rownum_per_period
                entity.input_rows = table.period_index
                entity.input_table = table.table
                entity.base_period = table.base_period

# this is what should happen
#                entity.indexed_input_table = entities_tables[ent_name]
#                entity.indexed_output_table = entities_tables[ent_name]

                # TODO: copying the table and generally preparing the output
                # file should be a different method than indexing
                print(" * copying table...")
                start_time = time.time()
                input_rows = entity.input_rows
                output_rows = dict((p, rows)
                                   for p, rows in input_rows.iteritems()
                                   if p < start_period)
                if output_rows:
                    # stoprow = last row of the last period before start_period
                    _, stoprow = input_rows[max(output_rows.iterkeys())]
                else:
                    stoprow = 0

                output_table = copy_table(table.table, output_entities,
                                          entity.fields.in_output.dtype,
                                          stop=stoprow,
                                          show_progress=True)
                entity.output_rows = output_rows
                print("done (%s elapsed)." % time2str(time.time() - start_time))

                print(" * building array for first simulated period...",
                      end=' ')
                start_time = time.time()

                # TODO: this whole process of merging all periods is very
                # opinionated and does not allow individuals to die/disappear
                # before the simulation starts. We couldn't for example,
                # take the output of one of our simulation and
                # re-simulate only some years in the middle, because the dead
                # would be brought back to life. In conclusion, it should be
                # optional.
                entity.array, entity.id_to_rownum = \
                    build_period_array(table.table,
                                       list(entity.fields.name_types),
                                       entity.input_rows,
                                       entity.input_index, start_period)
                assert isinstance(entity.array, ColumnArray)
                entity.array_period = start_period
                print("done (%s elapsed)." % time2str(time.time() - start_time))
                entity.table = output_table
        except:
            input_file.close()
            output_file.close()
            raise

        return input_file, output_file, dataset['globals']


class Void(DataSource):
    def __init__(self, output_path):
        self.output_path = output_path

    def run(self, globals_def, entities, start_period):
        globals_data = load_path_globals(globals_def)
        output_file = tables.open_file(self.output_path, mode="w")
        output_indexes = output_file.create_group("/", "indexes", "Indexes")
        output_entities = output_file.create_group("/", "entities", "Entities")
        for entity in entities.itervalues():
            entity.array = ColumnArray.empty(0, dtype=entity.fields.dtype)
            entity.array_period = start_period
            entity.id_to_rownum = np.empty(0, dtype=int)
            output_table = output_file.create_table(
                output_entities, entity.name, entity.fields.in_output.dtype,
                title="%s table" % entity.name)

            entity.input_table = None
            entity.table = output_table
            index_node = output_file.create_group(output_indexes, entity.name)
            entity.output_index_node = index_node
        return None, output_file, globals_data


def entities_from_h5(fpath):
    from entities import Entity
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
                global_def = normalize_type(table.dtype.type)
            else:
                global_def = get_fields(table)
            globals_def[table.name] = global_def
    h5in.close()
    return globals_def, entities
