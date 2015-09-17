# encoding: utf-8
from __future__ import print_function

import csv
import os.path
import re
from itertools import islice, chain

import numpy as np
try:
    import bcolz
except ImportError:
    bcolz = None
import tables
import yaml

from utils import (validate_dict, merge_dicts, merge_items, invert_dict,
                   countlines, skip_comment_cells, strip_rows, PrettyTable,
                   unique, duplicates, unique_duplicate, prod,
                   field_str_to_type, fields_yaml_to_type, fromiter,
                   LabeledArray)
from expr import missing_values


MB = 2.0 ** 20


def to_int(v):
    if not v or v == '--':
        return -1
    else:
        return int(v)

NaN = np.NaN


def to_float(v):
    if not v or v == '--':
        return NaN
    else:
        return float(v)


def to_bool(v):
    return v.lower() in ('1', 'true')

converters = {bool: to_bool,
              int: to_int,
              float: to_float,
              str: lambda v: v}


def convert(iterable, fields, positions=None):
    funcs = [converters[type_] for _, type_ in fields]
    if positions is None:
        for row in iterable:
            yield tuple(func(value) for func, value in zip(funcs, row))
    else:
        assert len(positions) <= len(fields)
        rowlen = None
        for row in iterable:
            if rowlen is None:
                rowlen = len(row)
            elif len(row) != rowlen:
                raise Exception("invalid row length (%d != %d): %s"
                                % (len(row), rowlen, row))

            # Note that [x for x in y] is generally faster than
            # tuple(x for x in y) but the stream is usually consumed by
            # fromiter which only accepts tuples (because a[i] = x works only
            # when x is a tuple)
            yield tuple(func(row[pos])
                        for func, pos in zip(funcs, positions))


def convert_2darray(iterable, celltype):
    """homogeneous 2d array"""

    func = converters[celltype]
    return [tuple(func(value) for value in row) for row in iterable]


def convert_1darray(iterable, celltype=None):
    """homogeneous 1d array"""
    if celltype is None:
        celltype = detect_column_type(iterable)
    func = converters[celltype]
    return [func(value) for value in iterable]


def guess_type(v):
    if not v or v == '--':
        return None

    v = v.lower()
    if v in ('0', '1', 'false', 'true'):
        return bool
    elif v.isdigit():
        return int
    else:
        try:
            float(v)
            return float
        except ValueError:
            return str


def detect_column_type(iterable):
    iterator = iter(iterable)
    coltype = 0
    type2code = {None: 0, bool: 1, int: 2, float: 3, str: 4}
    for value in iterator:
        coltype = max(coltype, type2code[guess_type(value)])
        if coltype == 4:
            break
    if coltype == 0:
        raise Exception("cannot detect column type (it is entirely empty)")
    return [None, bool, int, float, str][coltype]


# it is possible to express detect_column_types in terms of
# split_columns_as_iterators and detect_column_type, but it is both more
# complicated and slower, so let's keep this seemingly duplicate code here.
def detect_column_types(iterable):
    iterator = iter(iterable)
    header = iterator.next()
    numcolumns = len(header)
    coltypes = [0] * numcolumns
    type2code = {None: 0, bool: 1, int: 2, float: 3, str: 4}
    for row in iterator:
        if len(row) != numcolumns:
            raise Exception("all rows do not have the same number of columns")
        for column, value in enumerate(row):
            coltypes[column] = max(coltypes[column],
                                   type2code[guess_type(value)])
    for i, colname in enumerate(header):
        coltype = coltypes[i]
        if coltype == 0:
            print("Warning: column %s is all empty, assuming it is float"
                  % colname)
            coltypes[i] = 3
    num2type = [None, bool, int, float, str]
    return [(name, num2type[coltype])
            for name, coltype in zip(header, coltypes)]


def transpose_table(data):
    numrows = len(data)
    numcols = len(data[0])

    for rownum, row in enumerate(data, 1):
        if len(row) != numcols:
            raise Exception('line %d has %d columns instead of %d !'
                            % (rownum, len(row), numcols))

    return [[data[rownum][colnum] for rownum in range(numrows)]
            for colnum in range(numcols)]


def eval_with_template(s, template_context):
    return eval(s.format(**template_context), {'__builtins__': None})


class CSV(object):
    eval_re = re.compile('eval\((.*)\)')

    def __init__(self, fpath, newnames=None, delimiter=None, transpose=False):
        f = open(fpath, "rb")
        if delimiter is None:
            dialect = csv.Sniffer().sniff(f.read(1024))
#            dialect = csv.Sniffer().sniff(f.read(1024), ',:|\t')
            f.seek(0)
            data_stream = csv.reader(f, dialect)
        else:
            data_stream = csv.reader(f, delimiter=delimiter)
        if transpose:
            transposed = transpose_table(list(data_stream))
            data_stream = iter(transposed)
        else:
            transposed = None
        self.fpath = fpath
        if newnames is not None:
            # TODO: move this junk out of the class
            basename = os.path.splitext(os.path.basename(fpath))[0]
            for k in newnames:
                m = self.eval_re.match(newnames[k])
                if m:
                    eval_str = m.group(1)
                    newnames[k] = eval_with_template(eval_str,
                                                     {'basename': basename})
        self.newnames = newnames
        self.transposed = transposed
        self.f = f
        self.data_stream = data_stream
        self._fields = None
        self._field_names = None
        self._numlines = None

    def __iter__(self):
        return iter(self.data_stream)

    def next(self):
        return self.data_stream.next()

    def rewind(self):
        if self.transposed is not None:
            self.data_stream = iter(self.transposed)
        else:
            self.f.seek(0)

    def close(self):
        self.f.close()

    @property
    def field_names(self):
        if self._field_names is None:
            # TODO: use self._fields instead if it was already computed
            # read the first line in the file
            self.rewind()
            fnames = self.next()
            if self.newnames is not None:
                fnames = [self.newnames.get(name, name) for name in fnames]
            self._field_names = fnames

        return self._field_names

    @property
    def fields(self):
        if self._fields is None:
            self.rewind()
            fields = detect_column_types(self.data_stream)
            if self.newnames is not None:
                fields = [(self.newnames.get(name, name), type_)
                          for name, type_ in fields]
            self._fields = fields
        return self._fields

    @property
    def numlines(self):
        if self._numlines is None:
            if self.transposed is not None:
                self._numlines = len(self.transposed) - 1
            else:
                self._numlines = countlines(self.fpath) - 1
        return self._numlines

    def read(self, fields=None):
        """imports one Xsv file with all columns
           * columns can be in any order (they will be reordered if needed)
           * row order is preserved
        """
        print(" - reading", self.fpath)
        if fields is None:
            fields = self.fields
            positions = None
        else:
            available = self.field_names
            missing = set(name for name, _ in fields) - set(available)
            if missing:
                raise Exception("%s does not contain any field(s) named: %s"
                                % (self.fpath, ", ".join(missing)))
            positions = [available.index(name) for name, _ in fields]
        self.rewind()
        self.next()
        return convert(self.data_stream, fields, positions)

    def as_array(self, fields=None):
        if fields is None:
            fields = self.fields

        # csv file is assumed to be in the correct order (ie by period then id)
        datastream = self.read(fields)
        return fromiter(datastream, dtype=np.dtype(fields),
                        count=self.numlines)


def complete_path(prefix, path):
    """make a path absolute by prefixing it if necessary"""
    if os.path.isabs(path):
        return path
    else:
        return os.path.join(prefix, path)


def compression_str2filter(compression):
    if compression is not None:
        if '-' in compression:
            complib, complevel = compression.split('-')
            complevel = int(complevel)
        else:
            complib, complevel = compression, 5

        return ("(using %s level %d compression)" % (complib, complevel),
                tables.Filters(complevel=complevel, complib=complib))
    else:
        return "uncompressed", None


def stream_to_array(fields, datastream, numlines=None, invert=()):
    # make sure datastream is an iterator, not a list, otherwise it could
    # loop indefinitely as it will never be consumed.
    # Note that, contrary to what I thought, we shouldn't make a special case
    # for that as np.fromiter(islice(iter(l), max_rows)) is faster than
    # np.array(l[:max_rows])
    datastream = iter(datastream)
    dtype = np.dtype(fields)
    count = -1 if numlines is None else numlines
    array = fromiter(datastream, dtype=dtype, count=count)
    for field in invert:
        array[field] = ~array[field]
    return array


def stream_to_table(h5file, node, name, fields, datastream, numlines=None,
                    title=None, invert=(), buffersize=10 * 2 ** 20,
                    compression=None):
    # make sure datastream is an iterator, not a list, otherwise it could
    # loop indefinitely as it will never be consumed.
    # Note that, contrary to what I thought, we shouldn't make a special case
    # for that as np.fromiter(islice(iter(l), max_rows)) is faster than
    # np.array(l[:max_rows])
    datastream = iter(datastream)
    msg, filters = compression_str2filter(compression)
    print(" - storing %s..." % msg)
    dtype = np.dtype(fields)
    table = h5file.create_table(node, name, dtype, title=title, filters=filters)
    # buffered load
    max_buffer_rows = buffersize // dtype.itemsize
    while True:
        dataslice = islice(datastream, max_buffer_rows)
        if numlines is not None:
            if numlines <= 0:
                break
            buffer_rows = min(numlines, max_buffer_rows)
            # ideally, we should preallocate an empty buffer and reuse it,
            # but that does not seem to be supported by numpy
            array = fromiter(dataslice, dtype=dtype, count=buffer_rows)
            numlines -= buffer_rows
        else:
            array = fromiter(dataslice, dtype=dtype)
            if not len(array):
                break

        for field in invert:
            array[field] = ~array[field]
        table.append(array)
        table.flush()

    return table


def array_to_disk_array(node, name, array, title='', compression=None):
    h5file = node._v_file
    msg, filters = compression_str2filter(compression)
    print(" - storing %s..." % msg)
    if filters is not None:
        disk_array = h5file.create_carray(node, name, array, title,
                                          filters=filters)
    else:
        disk_array = h5file.create_array(node, name, array, title)
    if isinstance(array, LabeledArray):
        attrs = disk_array.attrs
        # pytables serialises Python lists as pickles but np.arrays as native
        # types, so it is more portable this way
        attrs.dimensions = np.array(array.dim_names)
        # attrs.dim0_pvalues = array([a, b, c])
        # attrs.dim1_pvalues = array([d, e])
        # ...
        for i, pvalues in enumerate(array.pvalues):
            setattr(attrs, 'dim%d_pvalues' % i, pvalues)
    return disk_array


def union1d(arrays):
    """arrays is an iterable returning arrays"""
    result = arrays.next()
    for array in arrays:
        result = np.union1d(result, array)
    return result


def interpolate(target, arrays, id_periods, fields):
    print(" * indexing...")
    periods = np.unique(id_periods['period'])
    max_id = np.max(id_periods['id'])

    row_for_id = {}
    for period in periods:
        # this might seem very wasteful but when compressed through
        # bcolz it is much smaller than an (id, rownum) dict, while
        # being only a bit slower
        row_for_id[period] = np.empty(max_id + 1, dtype=int)
        row_for_id[period].fill(-1)

    numrows = len(id_periods)
    lastrow_for_id = {}

    # compressing this with bcolz yield interesting compression but
    # is really too slow to use afterwards because access is
    # not sequential at all.
    nextrow_for_id = np.empty(numrows, dtype=int)
    nextrow_for_id.fill(-1)
    for rownum, (period, record_id) in enumerate(id_periods):
        row_for_id[period][record_id] = rownum

        # this assumes id_periods are ordered by period, which
        # is implicitly the case because of union1d
        lastrow = lastrow_for_id.get(record_id)
        if lastrow is not None:
            nextrow_for_id[lastrow] = rownum
        lastrow_for_id[record_id] = rownum
    del lastrow_for_id

    size = sum(row_for_id[period].nbytes for period in periods)

    if bcolz is not None:
        print(" * compressing index (%.2f Mb)..." % (size / MB), end=' ')
        for period in periods:
            row_for_id[period] = bcolz.carray(row_for_id[period])
        csize = sum(row_for_id[period].cbytes for period in periods)
        print("done. (%.2f Mb)" % (csize / MB))
    else:
        print('bcolz package not found (bcolz is required to use compression in interpolate)')

    print(" * interpolating...")
    for values in arrays:
        # sort it by id, then period
        values.sort(order=('id', 'period'))

        input_stream = iter(values)

        # eg we get:
        # 10, 2001, 5324.0
        # 10, 2004, 6200.0
        # 10, 2005, 7300.0
        prev_row = input_stream.next()
        fields_to_set = \
            [name for name in prev_row.dtype.names
             if name in target.dtype.names and name not in ('id', 'period')]
        fields_to_interpolate = [name for name in fields_to_set
                                 if name in fields]
        fields_to_copy = [name for name in fields_to_set
                          if name not in fields]
        # compute row in target array
        # 10, 2001, 5324.0 -> rowtofill = 154
        # 10, 2004, 6200.0
        # 10, 2005, 7300.0
        rowtofill = row_for_id[prev_row['period']][prev_row['id']]
        for row in input_stream:
            target_row = target[rowtofill]
            for fname in fields_to_copy:
                target_row[fname] = prev_row[fname]

            # 10, 2001, 5324.0 -> rowtofill = 154
            # 10, 2004, 6200.0 -> rownum = 180
            # 10, 2005, 7300.0
            rownum = row_for_id[row['period']][row['id']]
            # fill the row corresponding to the previous data point at
            # id, period (rowtofill) *and* all the rows (in the target
            # array) in between that point and this one (rownum)
            while rowtofill != -1 and rowtofill != rownum:
                target_row = target[rowtofill]
                for fname in fields_to_interpolate:
                    target_row[fname] = prev_row[fname]
                rowtofill = nextrow_for_id[rowtofill]

            rowtofill = rownum
            prev_row = row

        while rowtofill != -1:
            target_row = target[rowtofill]
            for fname in fields_to_interpolate:
                target_row[fname] = prev_row[fname]
            rowtofill = nextrow_for_id[rowtofill]


def load_ndarray(fpath, celltype=None):
    print(" - reading", fpath)
    with open(fpath, "rb") as f:
        reader = csv.reader(f)
        line_stream = skip_comment_cells(strip_rows(reader))
        header = line_stream.next()
        str_table = []
        for line in line_stream:
            if any(value == '' for value in line):
                raise Exception("empty cell found in %s" % fpath)
            str_table.append(line)
    ndim = len(header)

    # handle last dimension header (horizontal values)
    last_d_header = str_table.pop(0)
    # auto-detect type of values for the last d and convert them
    last_d_pvalues = convert_1darray(last_d_header)

    unique_last_d, dupe_last_d = unique_duplicate(last_d_pvalues)
    if dupe_last_d:
        print(("Duplicate column header value(s) (for '%s') in '%s': %s"
              % (header[-1], fpath,
                 ", ".join(str(v) for v in dupe_last_d))))
        raise Exception("bad data in '%s': found %d "
                        "duplicate column header value(s)"
                        % (fpath, len(dupe_last_d)))

    # handle other dimensions header

    # strip the ndim-1 first columns
    headers = [[line.pop(0) for line in str_table]
               for _ in range(ndim - 1)]
    headers = [convert_1darray(pvalues_str) for pvalues_str in headers]
    if ndim > 1:
        # having duplicate values is normal when there are more than 2
        # dimensions but we need to test whether there are duplicates of
        # combinations.
        dupe_combos = list(duplicates(zip(*headers)))
        if dupe_combos:
            print(("Duplicate row header value(s) in '%s':" % fpath))
            print((PrettyTable(dupe_combos)))
            raise Exception("bad alignment data in '%s': found %d "
                            "duplicate row header value(s)"
                            % (fpath, len(dupe_combos)))

    possible_values = [np.array(list(unique(pvalues))) for pvalues in headers]
    possible_values.append(np.array(unique_last_d))

    shape = tuple(len(values) for values in possible_values)
    num_possible_values = prod(shape)

    # transform the 2d table into a 1d list
    str_table = list(chain.from_iterable(str_table))
    if len(str_table) != num_possible_values:
        raise Exception("incoherent data in '%s': %d data cells "
                        "found while it should be %d based on the number "
                        "of possible values in headers (%s)"
                        % (fpath,
                           len(str_table),
                           num_possible_values,
                           ' * '.join(str(len(values))
                                      for values in possible_values)))

    # TODO: compare time with numpy built-in conversion:
    # if dtype is None, numpy tries to detect the best type itself
    # which it does a good job of if the values are already numeric values
    # if dtype is provided, numpy does a good job to convert from string
    # values.
    if celltype is None:
        celltype = detect_column_type(str_table)
    data = convert_1darray(str_table, celltype)
    array = np.array(data, dtype=celltype)
    return LabeledArray(array.reshape(shape), header, possible_values)


def load_def(localdir, ent_name, section_def, required_fields):
    if 'type' in section_def and 'fields' in section_def:
        raise Exception("invalid structure for '%s': "
                        "type and fields sections are mutually exclusive"
                        % ent_name)

    if 'type' in section_def:
        csv_filename = section_def.get('path', ent_name + ".csv")
        csv_filepath = complete_path(localdir, csv_filename)
        str_type = section_def['type']
        if isinstance(str_type, basestring):
            celltype = field_str_to_type(str_type, "array '%s'" % ent_name)
        else:
            assert isinstance(str_type, type)
            celltype = str_type
        return 'ndarray', load_ndarray(csv_filepath, celltype)

    fields_def = section_def.get('fields')
    if fields_def is not None:
        for fdef in fields_def:
            if isinstance(fdef, basestring):
                raise SyntaxError("invalid field declaration: '%s', you are "
                                  "probably missing a ':'" % fdef)
        if all(isinstance(fdef, dict) for fdef in fields_def):
            fields = fields_yaml_to_type(fields_def)
        else:
            assert all(isinstance(fdef, tuple) for fdef in fields_def)
            fields = fields_def
        fnames = {name for name, _ in fields}
        for reqname, reqtype in required_fields[::-1]:
            if reqname not in fnames:
                fields.insert(0, (reqname, reqtype))
    else:
        fields = None
    newnames = merge_dicts(invert_dict(section_def.get('oldnames', {})),
                           section_def.get('newnames', {}))
    transpose = section_def.get('transposed', False)

    interpolate_def = section_def.get('interpolate')
    files_def = section_def.get('files')
    if files_def is None:
        # XXX: it might be cleaner to use the same code path than for the
        # multi-file case (however, that would loose the "import any file
        # size" feature that I'm fond of.

        # we can simply return the stream as-is
        # FIXME: stream is not sorted
        # csv file is assumed to be in the correct order (ie by period then id)
        csv_filename = section_def.get('path', ent_name + ".csv")
        csv_filepath = complete_path(localdir, csv_filename)
        csv_file = CSV(csv_filepath, newnames,
                       delimiter=',', transpose=transpose)
        stream = csv_file.read(fields)
        if fields is None:
            fields = csv_file.fields
        if interpolate_def is not None:
            raise Exception('interpolate is currently only supported with '
                            'multiple files')
        return 'table', (fields, csv_file.numlines, stream, csv_file)
    else:
        # we have to load all files, merge them and return a stream out of that
        print(" * computing number of rows...")

        # 1) only load required fields
        default_args = dict(newnames=newnames, transpose=transpose)
        if isinstance(files_def, dict):
            files_items = files_def.items()
        elif isinstance(files_def, list) and files_def:
            if isinstance(files_def[0], dict):
                # handle YAML ordered dict structure
                files_items = [d.items()[0] for d in files_def]
            elif isinstance(files_def[0], basestring):
                files_items = [(path, {}) for path in files_def]
            else:
                raise Exception("invalid structure for 'files'")
        else:
            raise Exception("invalid structure for 'files'")

        # XXX: shouldn't we use the "path" defined for the whole entity if any?
        # section_def.get('path')
        files = []
        for path, kwargs in files_items:
            kwargs['newnames'] = \
                merge_dicts(invert_dict(kwargs.pop('oldnames', {})),
                            kwargs.get('newnames', {}))
            f = CSV(complete_path(localdir, path),
                    **merge_dicts(default_args, kwargs))
            files.append(f)
        id_periods = union1d(f.as_array(required_fields) for f in files)

        print(" * reading files...")
        # 2) load all fields
        if fields is None:
            target_fields = merge_items(*[f.fields for f in files])
            fields_per_file = [None for _ in files]
        else:
            target_fields = fields
            fields_per_file = [[(name, type_) for name, type_ in target_fields
                                if name in f.field_names]
                               for f in files]
            total_fields = set.union(*[set(f.field_names) for f in files])
            missing = set(name for name, _ in target_fields) - total_fields
            if missing:
                raise Exception("the following fields were not found in any "
                                "file: %s" % ", ".join(missing))

        total_lines = len(id_periods)

        # allocate main array
        target = np.empty(total_lines, dtype=np.dtype(target_fields))
        # fill with default values
        target[:] = tuple(missing_values[ftype] for _, ftype in target_fields)
        target['period'] = id_periods['period']
        target['id'] = id_periods['id']

        arrays = [f.as_array(fields_to_load)
                  for f, fields_to_load in zip(files, fields_per_file)]

        # close all files
        for f in files:
            f.close()

        # FIXME: interpolation currently only interpolates missing data points,
        # not data points with their value equal the missing value
        # corresponding to the field type. This can only be fixed once
        # booleans are loaded as int8.
        if interpolate_def is not None:
            if any(v != 'previous_value'
                   for v in interpolate_def.itervalues()):
                raise Exception("currently, only 'previous_value' "
                                "interpolation is supported")
            to_interpolate = [k for k, v in interpolate_def.iteritems()
                              if v == 'previous_value']
        else:
            to_interpolate = []

        interpolate(target, arrays, id_periods, to_interpolate)
        return 'table', (target_fields, total_lines, iter(target), None)


def csv2h5(fpath = None, buffersize=10 * 2 ** 20):
    with open(fpath) as f:
        content = yaml.load(f)

    yaml_layout = {
        '#output': str,
        'compression': str,
        'globals': {
            'periodic': {
                'path': str,
                'fields': [{
                    '*': str
                }],
                'oldnames': {
                    '*': str
                },
                'newnames': {
                    '*': str
                },
                'invert': [str],
                'transposed': bool
            },
            '*': {
                'path': str,
                'type': str,
                'fields': [{
                    '*': str
                }],
                'oldnames': {
                    '*': str
                },
                'newnames': {
                    '*': str
                },
                'invert': [str],
                'transposed': bool
            }
        },
        '#entities': {
            '*': {
                'path': str,
                'fields': [{
                    '*': str
                }],
                'oldnames': {
                    '*': str
                },
                'newnames': {
                    '*': str
                },
                'invert': [str],
                'transposed': bool,
                'files': None,
#                {
#                    '*': None
#                }
                'interpolate': {
                    '*': str
                }
            }
        }
    }

    validate_dict(content, yaml_layout)
    localdir = os.path.dirname(os.path.abspath(fpath))

    h5_filename = content['output']
    compression = content.get('compression')
    h5_filepath = complete_path(localdir, h5_filename)
    print("Importing in", h5_filepath)
    h5file = None
    try:
        h5file = tables.open_file(h5_filepath, mode="w", title="CSV import")

        globals_def = content.get('globals', {})
        if globals_def:
            print()
            print("globals")
            print("-------")
            const_node = h5file.create_group("/", "globals", "Globals")
            for global_name, global_def in globals_def.iteritems():
                print()
                print(" %s" % global_name)
                req_fields = ([('PERIOD', int)] if global_name == 'periodic'
                                                else [])

                kind, info = load_def(localdir, global_name,
                                      global_def, req_fields)
                if kind == 'ndarray':
                    array_to_disk_array(const_node, global_name, info,
                                        title=global_name,
                                        compression=compression)
                else:
                    assert kind == 'table'
                    fields, numlines, datastream, csvfile = info
                    stream_to_table(h5file, const_node, global_name, fields,
                                    datastream, numlines,
                                    title="%s table" % global_name,
                                    buffersize=buffersize,
                                    # FIXME: handle invert
                                    compression=compression)
                    if csvfile is not None:
                        csvfile.close()

        print()
        print("entities")
        print("--------")
        ent_node = h5file.create_group("/", "entities", "Entities")
        for ent_name, entity_def in content['entities'].iteritems():
            print()
            print(" %s" % ent_name)
            kind, info = load_def(localdir, ent_name,
                                  entity_def, [('period', int), ('id', int)])
            assert kind == "table"
            fields, numlines, datastream, csvfile = info

            stream_to_table(h5file, ent_node, ent_name, fields,
                            datastream, numlines,
                            title="%s table" % ent_name,
                            invert=entity_def.get('invert', []),
                            buffersize=buffersize, compression=compression)
            if csvfile is not None:
                csvfile.close()
    finally:
        if h5file is not None:
            h5file.close()
    print()
    print("done.")
