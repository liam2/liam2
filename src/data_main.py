import os
import csv
from itertools import islice

import numpy as np
import carray as ca
import tables
import yaml

from entities import entity_registry
from utils import timed
from properties import missing_values
from simulation import Simulation
        
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
    funcs = [converters[type_] for name, type_ in fields]
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
        assert len(row) == numcolumns, \
               "all rows do not have the same number of columns"
        for column, value in enumerate(row):
            coltypes[column] = max(coltypes[column],
                                   type2code[guess_type(value)])
    for colname, coltype in zip(header, coltypes):
        assert coltype != 0, "column %s is all empty" % colname
    num2type = [None, bool, int, float]
    return [(name, num2type[coltypes[column]])
            for column, name in enumerate(header)]

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
    
def import_csv(fpath, fields=None, delimiter=",", transpose=False):
    '''imports one Xsv file with all columns 
       time, id, value1, value2, value3 (not necessarily in that order)
    '''
    print " * reading", fpath

    f = open(fpath, "rb")
    data_stream = csv.reader(f, delimiter=delimiter)
    if transpose:
        transposed = transpose_table(list(data_stream))
        data_stream = iter(transposed)
        
    if fields is None:
        fields = detect_column_types(data_stream)
        
        # rewind the stream, as it was consumed by the detection
        if transpose:
            data_stream = iter(transposed)
        else:
            f.seek(0)
        # skip header
        data_stream.next()
            
        positions = None
    else:
        header = data_stream.next()
        fieldnames = [name for name, _ in fields]
        missing_columns = set(fieldnames) - set(header)
        assert not missing_columns, "missing field(s): %s" % \
               ", ".join(missing_columns)
    
        positions = [header.index(fieldname) for fieldname in fieldnames]
    return fields, convert_stream_and_close_file(f, data_stream, fields,
                                                 positions)         

def countlines(filepath):
    with open(filepath) as f:
        return sum(1 for _ in f)

def complete_path(prefix, directory):
    if os.path.isabs(directory):
        return directory
    else:
        return os.path.join(prefix, directory) 

def stream_to_table(h5file, node, name, fields, datastream, numlines=None,
                    title=None, invert=(), buffersize=10 * 2 ** 20,
                    compression=None):
    if compression is not None:
        if '-' in compression:
            complib, complevel = compression.split('-')
            complevel = int(complevel)
        else:
            complib, complevel = compression, 5
    
        print " * storing (using %s level %d compression)..." \
              % (complib, complevel)
        filters = tables.Filters(complevel=complevel, complib=complib)
    else:
        print " * storing uncompressed..."
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
            buffer = np.fromiter(dataslice, dtype=dtype, count=buffer_rows)
            numlines -= buffer_rows
        else:
            buffer = np.fromiter(dataslice, dtype=dtype)
            if not len(buffer):
                break
        
        for field in invert:
            buffer[field] = ~buffer[field]
        table.append(buffer)
        
        table.flush()

    return table

def load_def(localdir, ent_name, section_def, required_fields):
    csv_filename = section_def.get('path', ent_name + ".csv")
    csv_filepath = complete_path(localdir, csv_filename)
    oldnames = section_def.get('oldnames')
    transpose = section_def.get('transposed', False)
    fields_def = section_def.get('fields')
    if fields_def is not None:
        # handle YAML ordered dict structure
        field_list = [d.items()[0] for d in fields_def]
        # convert string type to real types
        fields = [(name, str_to_type[type_]) for name, type_ in field_list]
        fields[0:0] = required_fields
        if oldnames is None:
            fields_oldnames = fields
        else:
            fields_oldnames = [(oldnames.get(name, name), type_)
                               for name, type_ in fields]
    else:
        fields_oldnames = None

    numlines = countlines(csv_filepath) - 1
    fields_oldnames, datastream = import_csv(csv_filepath, fields_oldnames,
                                             transpose=transpose)
    
    if fields_def is None:
        if oldnames is None:
            fields = fields_oldnames
        else:
            newnames = dict((v, k) for k, v in oldnames.iteritems())
            fields = [(newnames.get(name, name), type_)
                      for name, type_ in fields_oldnames]
        #TODO: two different messages if field missing or field has a bad type
        for name, type_ in required_fields:
            assert (name, type_) in fields, \
                   "%s does not contain a '%s' field of type '%s'. You " \
                   "should either rename the corresponding field or use " \
                   "'oldnames'." % (csv_filename, name, type_.__name__)  
    return fields, numlines, datastream

    
def csv2h5(fpath, buffersize=10 * 2 ** 20):
    with open(fpath) as f:
        content = yaml.load(f)
                    
    localdir = os.path.dirname(os.path.abspath(fpath))

    h5_filename = content['output']
    compression = content.get('compression')
    h5_filepath = complete_path(localdir, h5_filename)
    print "Importing in", h5_filepath
    h5file = tables.openFile(h5_filepath, mode="w", title="CSV import")
    
    periodic_def = content.get('globals', {}).get('periodic')
    if periodic_def is not None:
        print "*** globals ***"
        fields, numlines, datastream = load_def(localdir, "periodic",
                                                periodic_def, [('PERIOD', int)])
        const_node = h5file.createGroup("/", "globals", "Globals")
        stream_to_table(h5file, const_node, "periodic", fields, datastream,
                        title="Global time series",
                        compression=compression)
        
    ent_node = h5file.createGroup("/", "entities", "Entities")
    for ent_name, entity_def in content['entities'].iteritems():
        print "*** %s ***" % ent_name
        fields, numlines, datastream = load_def(localdir, ent_name, entity_def,
                                                [('period', int), ('id', int)])
        stream_to_table(h5file, ent_node, ent_name, fields, datastream,
                        numlines, title="%s table" % ent_name,
                        invert=entity_def.get('invert', []),
                        compression=compression)

    h5file.close()
    print "done."


###############
# LEGACY CODE #
###############

def export_csv_3col(file_path, data, colname, delimiter=","):
    with open(file_path, "wb") as f:
        dataWriter = csv.writer(f, delimiter=delimiter)
        dataWriter.writerow(["id", "time", "value"])
        for line in data:
            dataWriter.writerow([line["id"], line["period"], line[colname]]) 

def export_csv(file_path, data, colnames, delimiter=","):
    with open(file_path, "wb") as f:
        dataWriter = csv.writer(f, delimiter=delimiter)
        dataWriter.writerow(colnames)
        dataWriter.writerows(data)

def index_first(header, possible_values):
    values_present = set(header) & set(possible_values)
    assert len(values_present) == 1, "%s not found in header: %s" % (
                                     possible_values, header)
    colname = values_present.pop()
    return header.index(colname)

def import_csv_3col(fpath, vname, vtype, delimiter=","):
    '''imports one Xsv file which use a three columns format: 
       id, time, value (not necessarily in that order)
    ''' 
    with open(fpath, "rb") as f:
        dataReader = csv.reader(f, delimiter=delimiter)
        
        header = dataReader.next()
        id_pos = index_first(header, ['id', 'idnum', 'oid'])
        time_pos = index_first(header, ['time', 'year', 'period'])
        if vname is not None:
            func = converters[vtype]
            value_pos = index_first(header, ['value', 'did', vname])
            for row in dataReader:
                yield (int(row[id_pos]), int(row[time_pos]), func(row[value_pos]))
        else:
            for row in dataReader:
                yield (int(row[time_pos]), int(row[id_pos]))

# initially copied from simulation_txt2yaml
def load_av_globals(input_path):
    # macro.av is a csv with tabs OR spaces as separator and a header of 1
    # line
    with open(input_path, "rb") as f:
        lines = [line.split() for line in f.read().splitlines()]

    # eg: "sample 1955Y1 2060Y1"
    firstline = lines.pop(0)
    assert firstline[0] == "sample"
    def year_str2int(s):
        return int(s.replace('Y1', ''))
    start, stop = year_str2int(firstline[1]), year_str2int(firstline[2])
    num_periods = stop - start + 1

    transposed = transpose_table(lines)
    fields = detect_column_types(transposed)
    data = list(convert(transposed[1:], fields))
    
    assert fields[0] == ('YEAR', int)

    # rename YEAR to period
    fields[0] = ('period', int)
    assert len(data) == num_periods
    return fields, data

def load_csv_globals(input_path, fields=None, transpose=False):
    fields, data = import_csv(input_path, fields, transpose=transpose)
    
    assert ('period', int) in fields or ('PERIOD', int) in fields
    try:
        pos = fields.index(('PERIOD', int))
        # make a copy so that we don't modify the list instance we passed to the
        # import_csv generator (which will be executed only when the data is
        # actually used)
        fields = fields[:]
        fields[pos] = ('period', int)
    except ValueError:
        pass
    return fields, data

class ImportExportData(object):
    def __init__(self, simulation_fpath, io_fpath):
        self.simulation = Simulation(simulation_fpath)

        with open(io_fpath) as f:
            content = yaml.load(f)
                    
        import_export_def = content['import_export']
        
        self.h5 = self.simulation.input_path
        self.globals_path = import_export_def['globals_path']
        self.csv_path = import_export_def['csv_path']
        self.first_period = import_export_def.get('first_period')
        self.last_period = import_export_def.get('last_period')
        self.oldnames = import_export_def.get('oldnames', {})
        self.toinvert = import_export_def.get('invert', {})

    def read_table(self, table):
        if self.first_period is not None and \
           self.last_period is not None:
            return table.where("(period >= %d) & (period <= %d)" 
                               % (self.first_period, self.last_period))
        elif self.first_period is not None:
            return table.where("period >= %d" % self.first_period)
        elif self.last_period is not None:
            return table.where("period <= %d" % self.last_period)
        else:
            return table.iterrows()
        
    def export_tsv_3col(self, delimiter="\t"):
        print "Exporting from", self.h5
        h5file = tables.openFile(self.h5, mode="r")
        for ent_name in entity_registry:
            # = h5file.root.person if ent_name="person"
            table = getattr(h5file.root, ent_name)
            tab_fields = entity_registry[ent_name].fields
            to_export = [vname for vname, _ in tab_fields
                         if vname not in ("id", "period")]
            for vname in to_export:
                file_name = "%s_%s.txt" % (ent_name[0], vname)
                file_path = os.path.join(self.csv_path, file_name)
                print "writing", file_path
                array = self.read_table(table)
                export_csv_3col(file_path, array, vname, delimiter)
        h5file.close()
        
    def export_csv(self, delimiter=","):
        print "Exporting from", self.h5
        h5file = tables.openFile(self.h5, mode="r")
        entities = h5file.root.entities
        for ent_name in entity_registry:
            table = getattr(entities, ent_name)
            fields = [name for name, _ in entity_registry[ent_name].fields]
            file_name = "%s.csv" % ent_name
            file_path = os.path.join(self.csv_path, file_name)
            print "writing", file_path
            array = self.read_table(table)
            export_csv(file_path, array, fields, delimiter)
        h5file.close()
        
    def import_tsv(self, complib=None, complevel=5, delimiter="\t"):
        print "Importing in", self.h5
        print "*** globals ***"
        input_path = self.globals_path
        print "reading", input_path
        if input_path.endswith('.av'):
            global_fields, global_data = load_av_globals(input_path)
        elif input_path.endswith('.csv'):
            global_fields, global_data = load_csv_globals(input_path, transpose=True)
            
        h5file = tables.openFile(self.h5, mode="w", title="CSV import")
        missing_files = []

        print "building list of needed files..."
        to_load = {}
        for ent_name, entity in entity_registry.iteritems():
            oldnames = self.oldnames.get(ent_name, {})
            neededfields = [(name, type_) for name, type_ in entity.fields
                            if name not in set(entity.missing_fields)]

            many2one_links = [v for v in entity.links.itervalues()
                              if v._link_type == 'many2one']
            fields_toskip = set([link._link_field for link in many2one_links])
            fields_toskip.update(["id", "period"])
            
            entity_toload = []
            # list "fields" to be loaded
            for vname, vtype in neededfields:
                if vname not in fields_toskip:
                    oldname = oldnames.get(vname, vname)
                    filename = "%s_%s.txt" % (ent_name[0], oldname)
                    entity_toload.append((filename, vname, oldname, vtype))
            # list "links" to be loaded
            for link in many2one_links:
                vname = link._link_field
                oldname = oldnames.get(vname, vname)
                filename = "link_%s.txt" % link._name
                entity_toload.append((filename, vname, oldname, int))

            to_load[ent_name] = entity_toload
                
            for filename, _, _, _ in entity_toload:
                if not os.path.exists(os.path.join(self.csv_path, filename)):
                    missing_files.append(filename)

        if missing_files:
            raise Exception("missing files:\n - %s" 
                            % "\n - ".join(missing_files))

        const_node = h5file.createGroup("/", "globals", "Globals")
        stream_to_table(h5file, const_node, "periodic", global_fields,
                        global_data, title="Global time series")

        entities_node = h5file.createGroup("/", "entities", "Entities")

        #TODO: I need to work in 3 passes:
        # 1) compute global index
        # 2) for each field, complete missing rows and save the result to a binary file
        # 3) construct one large array out of it 

        # =========================
        # 1) read the files a first time to build the list of unique (period, id)
        #    and unique ids (to replace objtype_XX) 
        # 2) allocate the array (or a chunk of it) (if chunksize != lastchunk)
        # 2bis) fill it with default values?
        # 3) read each file and complete the array (or a chunk of it)
        # 4) flush data to hdf
        # 5) if using chunks goto 2
        # ============================
        
        # iterate over the the large index, and fill the large vector
        # as we go
        # keep current/last_value for the column in a vector the length = the number of unique id

        # 1 min 42 - 355/376Mb with dict[period][id]
        # 1 min 42 - 444/476Mb with dict[id][period]
        # 1 min 45 - 450/485Mb with dict[id] = (dict[period], list)
        # 1 min 51 - 320/340Mb nextrow_for_id
        # 2 min 04 - ~230/250Mb nextrow_for_id - compressed dict[period] = array 
        
        # since I want the result to be sorted by period, I could dump to 
        # disk the index of all periods except the current one  
        for ent_name, entity in entity_registry.iteritems():
            print "*** %s ***" % ent_name 
            toinvert = self.toinvert.get(ent_name, [])

            neededfields = [(name, type_) for name, type_ in entity.fields
                            if name not in set(entity.missing_fields)]

            toload_entity = to_load[ent_name]

            print " * computing number of rows..."
            id_periods = None
            id_periods_dtype = np.dtype([('period', int), ('id', int)])
            for filename, _, _, _ in toload_entity:
                print "   - reading", filename, "...",
                file_path = os.path.join(self.csv_path, filename)
                #TODO: check if it is faster in 2 passes (compute number of
                # lines first)
                numlines = countlines(file_path) - 1
                id_period_stream = import_csv_3col(file_path, None, None,
                                                   delimiter)
                file_id_periods = np.fromiter(id_period_stream,
                                              dtype=id_periods_dtype,
                                              count=numlines)
                if id_periods is None:
                    id_periods = file_id_periods
                else:
                    print "computing union ...",
                    id_periods = np.union1d(id_periods, file_id_periods)
                print "done."
                    
            print "   - unique periods..."
            periods = np.unique(id_periods['period'])
            print "   - unique ids..."
            ids = np.unique(id_periods['id'])
            max_id = np.max(ids)
            del ids
                 
            print " * indexing..."
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
            for rownum, (period, id) in enumerate(id_periods):
                row_for_id[period][id] = rownum
                
                # this assumes id_periods are ordered by period, which
                # is implicitly the case because of union1d
                lastrow = lastrow_for_id.get(id)
                if lastrow is not None:
                    nextrow_for_id[lastrow] = rownum
                lastrow_for_id[id] = rownum
            del lastrow_for_id

            size = sum(row_for_id[period].nbytes for period in periods)
            print " * compressing index (%.2f Mb)..." % (size / MB),
            for period in periods:
                row_for_id[period] = ca.carray(row_for_id[period])
            csize = sum(row_for_id[period].cbytes for period in periods)
            print "done. (%.2f Mb)" % (csize / MB)

            main_dtype = np.dtype(neededfields)
            print " * allocating main array (%d rows * %d bytes = %.2f Mb)..." \
                  % (numrows, main_dtype.itemsize, 
                     main_dtype.itemsize * numrows / MB)
            array = np.empty(numrows, dtype=main_dtype)

            fdescs = [(fname, missing_values[ftype])
                      for fname, ftype in neededfields]
            print " * filling with default values..."
            array[:] = tuple(default for fname, default in fdescs)
            array['period'] = id_periods['period']
            array['id'] = id_periods['id']
            
            del id_periods

            print " * loading..."
            for filename, vname, oldname, vtype in toload_entity:
                file_path = os.path.join(self.csv_path, filename)
                print "   - reading", filename, "(%s)" % vname, "...",

                values_stream = import_csv_3col(file_path, oldname, vtype,
                                                delimiter)
                dt = np.dtype([('id', int), ('period', int), ('value', vtype)])
                values = np.fromiter(values_stream, dtype=dt)
                print "sorting...",
                values.sort()
                
                print "transferring and completing...",
                prev_id, prev_period, prev_value = values[0]
                
                rowtofill = row_for_id[prev_period][prev_id]
                target = array[vname]
                for id, period, value in values[1:]:
                    rownum = row_for_id[period][id]
                    while rowtofill != -1 and rowtofill != rownum:  
                        target[rowtofill] = prev_value
                        rowtofill = nextrow_for_id[rowtofill]
                    rowtofill = rownum
                    prev_value = value
                    
                while rowtofill != -1:  
                    target[rowtofill] = prev_value
                    rowtofill = nextrow_for_id[rowtofill]
                print "done."

            if toinvert:
                print " * inverting..."
                for field in toinvert:
                    array[field] = ~array[field]

            if complib is not None:
                print " * storing (using %s level %d compression)..." \
                      % (complib, complevel)
                filters = tables.Filters(complevel=complevel, complib=complib)
            else:
                print " * storing uncompressed..."
                filters = None
            table = h5file.createTable(entities_node, entity.name, main_dtype,
                                       title="%s table" % entity.name,
                                       filters=filters)
            table.append(array)
            table.flush()

#            max_id = np.max(ids)
#            id_to_rownum = np.empty(max_id + 1, dtype=int)
#            id_to_rownum.fill(-1)
#            for rownum, id in enumerate(ids):
#                id_to_rownum[id] = rownum 
#            entity.id_to_rownum = id_to_rownum

        # this is done so that id_to_rownum arrays are smaller
#        print "shifting ids and links..."
#        for ent_name, entity in entity_registry.iteritems():
#            to_shift = [(l._link_field, l._target_entity) 
#                        for l in entity.links.values()
#                        if l._link_type == 'many2one']
#            to_shift.append(('id', ent_name))
#            print " * shifting %s fields:" % ent_name
#            for field, target_name in to_shift:
#                print "   -", field
#                target = entity_registry[target_name]
#                array = entity.array[field]
#                missing_int = missing_values[int]
#                id_to_rownum = target.id_to_rownum
#                entity.array[field] = np.where(array == missing_int,
#                                               missing_int, 
#                                               id_to_rownum[array])

        h5file.close()

if __name__ == '__main__':
    import sys

    print 'Using Python %s' % sys.version
    args = sys.argv

    valid_cmds = ('export_tsv', 'export_csv', 'import_tsv')
    if len(args) < 4:
        print """
Usage: %s action simulation_file import_export_file [extra1] [extra2] ...
  action can be any of %s
  for import_tsv, you can specify optional compression by two extra arguments:
  extra1 can be any of zlib, lzo, bzip2 or blosc (defaults to none)
  extra2 is the compression level: from 1 to 9 (defaults to 5)
""" % (args[0], ', '.join(repr(cmd) for cmd in valid_cmds))
        sys.exit()
    cmd = args[1]
    
    print "using simulation file: '%s' / '%s'" % (args[2], args[3])
    data = ImportExportData(args[2], args[3])

    assert cmd in valid_cmds, "Unknown command '%s'" % cmd
    timed(getattr(data, cmd), *args[4:])
