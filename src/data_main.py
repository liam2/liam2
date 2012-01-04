import os.path
import re
import csv
from itertools import islice

import numpy as np
import carray as ca
import tables
import yaml

from utils import validate_dict, merge_dicts, merge_items, invert_dict, \
                  countlines
from properties import missing_values

MB = 2.0 ** 20
str_to_type = {'float': float, 'int': int, 'bool': bool}


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
              float: to_float}


def convert(iterable, fields, positions=None):
    funcs = [converters[type_] for _, type_ in fields]
    if positions is None:
        for row in iterable:
            yield tuple(func(value) for func, value in zip(funcs, row))
    else:
        assert len(positions) <= len(fields)
        for row in iterable:
            yield tuple(func(row[pos])
                        for func, pos in zip(funcs, positions))


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
            raise ValueError("cannot determine type for '%s'" % v)


def detect_column_types(iterable):
    iterator = iter(iterable)
    header = iterator.next()
    numcolumns = len(header)
    coltypes = [0] * numcolumns
    type2code = {None: 0, bool: 1, int: 2, float: 3}
    for row in iterator:
        if len(row) != numcolumns:
            raise Exception("all rows do not have the same number of columns")
        for column, value in enumerate(row):
            coltypes[column] = max(coltypes[column],
                                   type2code[guess_type(value)])
    for colname, coltype in zip(header, coltypes):
        if coltype == 0:
            raise Exception("column %s is all empty" % colname)
    num2type = [None, bool, int, float]
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


def convert_stream_and_close_file(f, data_stream, fields, positions):
    #XXX: is there no way to avoid this? yield from?
    for row in convert(data_stream, fields, positions):
        yield row
    f.close()


def eval_with_template(s, template_context):
    return eval(s.format(**template_context), {'__builtins__': None})


class CSV(object):
    eval_re = re.compile('eval\((.*)\)')

    def __init__(self, fpath, newnames=None, delimiter=None, transpose=False):
        f = open(fpath, "rb")
        if delimiter is None:
            dialect = csv.Sniffer().sniff(f.read(1024))
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

        # performance hack
        self.next = self.data_stream.next

    def __iter__(self):
        return iter(self.data_stream)

#    def next(self):
#        return self.data_stream.next()

    def rewind(self):
        if self.transposed is not None:
            self.data_stream = iter(self.transposed)
            self.next = self.data_stream.next
        else:
            self.f.seek(0)

    def close(self):
        self.f.close()

    @property
    def field_names(self):
        if self._field_names is None:
            #TODO: use self._fields instead if it was already computed
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
            self._numlines = countlines(self.fpath) - 1
        return self._numlines

    def read(self, fields=None):
        '''imports one Xsv file with all columns
           * columns can be in any order (they will be reordered if needed)
           * row order is preserved
        '''
        print "  - reading", self.fpath
        if fields is None:
            fields = self.fields
            positions = None
        else:
            self.check_has_fields(fields)
            available_fields = self.field_names
            positions = [available_fields.index(name)
                         for name, _ in fields]
        self.rewind()
        self.next()
        return convert_stream_and_close_file(self.f, self.data_stream,
                                             fields, positions)
#        for row in convert(self.data_stream, fields, positions):
#            yield row
#        self.f.close()

    def check_has_fields(self, fields):
        missing_fields = set(name for name, _ in fields) - \
                             set(self.field_names)
        if missing_fields:
            raise Exception("%s does not contain any field(s) named: %s"
                            % (self.fpath, ", ".join(missing_fields)))

    def as_array(self, fields=None):
        if fields is None:
            fields = self.fields
        else:
            # we need to explicitly check for missing fields here, even if
            # it is already done in self.read() because np.fromiter
            # swallows exceptions, so the error message in that case is awful.
            self.check_has_fields(fields)

        # csv file is assumed to be in the correct order (ie by period then id)
        datastream = self.read(fields)
        return np.fromiter(datastream, dtype=np.dtype(fields),
                           count=self.numlines)


def complete_path(prefix, path):
    '''make a path absolute by prefixing it if necessary'''
    if os.path.isabs(path):
        return path
    else:
        return os.path.join(prefix, path)


def stream_to_table(h5file, node, name, fields, datastream, numlines=None,
                    title=None, invert=(), buffersize=10 * 2 ** 20,
                    compression=None):
    # make sure datastream is an iterator, not a list, otherwise it could
    # loop indefinitely as it will never be consumed.
    # Note that, contrary to what I thought, we shouldn't make a special case
    # for that as np.fromiter(islice(iter(l), max_rows)) is faster than
    # np.array(l[:max_rows])
    datastream = iter(datastream)

    if compression is not None:
        if '-' in compression:
            complib, complevel = compression.split('-')
            complevel = int(complevel)
        else:
            complib, complevel = compression, 5

        print "  - storing (using %s level %d compression)..." \
              % (complib, complevel)
        filters = tables.Filters(complevel=complevel, complib=complib)
    else:
        print "  - storing uncompressed..."
        filters = None
    dtype = np.dtype(fields)
    table = h5file.createTable(node, name, dtype, title=title, filters=filters)
    # buffered load
    max_buffer_rows = buffersize / dtype.itemsize
    while True:
        dataslice = islice(datastream, max_buffer_rows)
        if numlines is not None:
            if numlines <= 0:
                break
            buffer_rows = min(numlines, max_buffer_rows)
            # ideally, we should preallocate an empty buffer and reuse it,
            # but that does not seem to be supported by numpy
            array = np.fromiter(dataslice, dtype=dtype, count=buffer_rows)
            numlines -= buffer_rows
        else:
            array = np.fromiter(dataslice, dtype=dtype)
            if not len(array):
                break

        for field in invert:
            array[field] = ~array[field]
        table.append(array)
        table.flush()

    return table


def union1d(arrays):
    '''arrays is an iterable returning arrays'''
    result = arrays.next()
    for array in arrays:
        result = np.union1d(result, array)
    return result


def interpolate(target, arrays, id_periods, fields):
    print " * indexing..."
    periods = np.unique(id_periods['period'])
    max_id = np.max(id_periods['id'])

    row_for_id = {}
    for period in periods:
        # this might seem very wasteful but when compressed through
        # carray it is much smaller than an (id, rownum) dict, while
        # being only a bit slower
        row_for_id[period] = np.empty(max_id + 1, dtype=int)
        row_for_id[period].fill(-1)

    numrows = len(id_periods)
    lastrow_for_id = {}

    # compressing this with carray yield interesting compression but
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
    print " * compressing index (%.2f Mb)..." % (size / MB),
    for period in periods:
        row_for_id[period] = ca.carray(row_for_id[period])
    csize = sum(row_for_id[period].cbytes for period in periods)
    print "done. (%.2f Mb)" % (csize / MB)

    print " * loading and interpolating..."
    for values in arrays:
        # sort it by id, then period
        values.sort(order=('id', 'period'))

        input_stream = iter(values)

        # eg we get:
        # 10, 2001, 5324.0
        # 10, 2004, 6200.0
        # 10, 2005, 7300.0
        prev_row = input_stream.next()
        fields_to_set = [name for name in prev_row.dtype.names
                                 if name in target.dtype.names and
                                    name not in ('id', 'period')]
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


def load_def(localdir, ent_name, section_def, required_fields):
    fields_def = section_def.get('fields')
    if fields_def is not None:
        # handle YAML ordered dict structure
        for fdef in fields_def:
            if isinstance(fdef, basestring):
                raise SyntaxError("invalid field declaration: '%s', you are "
                                  "probably missing a ':'" % fdef)
        field_list = [d.items()[0] for d in fields_def]
        # convert string type to real types
        types = [(k, v) if isinstance(v, basestring) else (k, v['type'])
                 for (k, v) in field_list]
        for name, type_ in types:
            if type_ not in str_to_type:
                raise SyntaxError("'%s' is not a valid field type for field "
                                  "'%s'." % (type_, name))

        fields = [(name, str_to_type[type_]) for name, type_ in types]
    else:
        fields = None

    newnames = merge_dicts(invert_dict(section_def.get('oldnames', {})),
                           section_def.get('newnames', {}))
    transpose = section_def.get('transposed', False)

    interpolate_def = section_def.get('interpolate')
    files_def = section_def.get('files')
    if files_def is None:
        #XXX: it might be cleaner to use the same code path than for the
        # multi-file case (however, that would loose the "import any file
        # size" feature that I'm fond of.

        # we can simply return the stream as-is
        #FIXME: stream is not sorted
        # csv file is assumed to be in the correct order (ie by period then id)
        csv_filename = section_def.get('path', ent_name + ".csv")
        csv_filepath = complete_path(localdir, csv_filename)
        csv_file = CSV(csv_filepath, newnames,
                       delimiter=',', transpose=transpose)
        if fields is not None:
            fields = required_fields + fields
            # we have to check explicitly for missing fields here, even if it
            # is done in csv_file.read because of that stupid bug in
            # np.fromiter which swallows exceptions in generators
            csv_file.check_has_fields(fields)
        stream = csv_file.read(fields)
        if fields is None:
            fields = csv_file.fields
        if interpolate_def is not None:
            raise Exception('interpolate is currently only supported with '
                            'multiple files')
        return fields, csv_file.numlines, stream
    else:
        # we have to load all files, merge them and return a stream out of that
        print " * computing number of rows..."

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

        files = []
        for path, kwargs in files_items:
            kwargs['newnames'] = \
                merge_dicts(invert_dict(kwargs.pop('oldnames', {})),
                            kwargs.get('newnames', {}))
            f = CSV(complete_path(localdir, path),
                    **merge_dicts(default_args, kwargs))
            files.append(f)
        id_periods = union1d(f.as_array(required_fields) for f in files)

        # 2) load all fields
        if fields is None:
            target_fields = merge_items(*[f.fields for f in files])
            fields_per_file = [None for f in files]
        else:
            target_fields = required_fields + fields
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

        arrays = (f.as_array(fields_to_load)
                  for f, fields_to_load in zip(files, fields_per_file))

        #FIXME: interpolation currently only interpolates missing data points,
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
        return target_fields, total_lines, iter(target)


def csv2h5(fpath, buffersize=10 * 2 ** 20):
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
    print "Importing in", h5_filepath
    try:
        h5file = tables.openFile(h5_filepath, mode="w", title="CSV import")

        globals_def = content.get('globals', {})
        periodic_def = globals_def.get('periodic')
        if periodic_def is not None:
            print "* globals"
            fields, numlines, datastream = load_def(
                localdir, "periodic", periodic_def, [('PERIOD', int)])
            const_node = h5file.createGroup("/", "globals", "Globals")
            stream_to_table(h5file, const_node, "periodic", fields, datastream,
                            title="Global time series",
                            buffersize=buffersize, compression=compression)

        ent_node = h5file.createGroup("/", "entities", "Entities")
        for ent_name, entity_def in content['entities'].iteritems():
            print "* %s" % ent_name
            fields, numlines, datastream = load_def(
                localdir, ent_name, entity_def, [('period', int), ('id', int)])
            stream_to_table(h5file, ent_node, ent_name, fields, datastream,
                            numlines, title="%s table" % ent_name,
                            invert=entity_def.get('invert', []),
                            buffersize=buffersize, compression=compression)
    finally:
        h5file.close()
    print "done."
