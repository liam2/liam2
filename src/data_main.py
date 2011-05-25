import os
import csv
from itertools import izip, islice

import numpy as np
import carray as ca
import tables
import yaml

from entities import entity_registry
from utils import timed
from properties import missing_values
from simulation import Simulation
        
MB = 2.0 ** 20


def to_int(v):
    return int(float(v)) if v else 0

def to_float(v):
    return float(v) if v else 0.0

def to_bool(v):
    return v.lower() in ('1', 'true')

converters = {bool: to_bool,
              int: to_int,
              float: to_float}


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


def import_csv(fpath, fields, delimiter=","):
    '''imports one Xsv file with all columns 
       time, id, value1, value2, value3 (not necessarily in that order)
    '''
    print "reading", fpath

    with open(fpath, "rb") as f:
        dataReader = csv.reader(f, delimiter=delimiter)
        
        header = dataReader.next()
        
        fieldnames = [name for name, type_ in fields]
        missing_columns = set(fieldnames) - set(header)
        assert not missing_columns, "Missing fields: %s" % \
               ", ".join(missing_columns)
        
        positions = [header.index(fieldname) for fieldname in fieldnames]
        funcs = [converters[type_] for name, type_ in fields]
        for row in dataReader:
            yield tuple(func(row[pos]) for func, pos in zip(funcs, positions))


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
    
def transpose_table(data):
    numrows = len(data)
    numcols = len(data[0])
    
    for rownum, row in enumerate(data, 1):
        if len(row) != numcols:
            raise Exception('line %d has %d columns instead of %d !'
                            % (rownum, len(row), numcols))
    
    return [[data[rownum][colnum] for rownum in range(numrows)]
            for colnum in range(numcols)]

def transpose_and_convert(lines):
    transposed = transpose_table(lines)
    names = transposed.pop(0)
    funcs = [float for _ in range(len(lines))]
    funcs[0] = int
    converted = [tuple([func(cell.replace('--', 'NaN'))
                  for cell, func in izip(row, funcs)])
                 for row in transposed]
    return names, converted


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
        
    # copied from simulation_txt2yaml
    def load_av_globals(self, input_path):
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

        names, data = transpose_and_convert(lines)
        assert names[0] == 'YEAR'
        # rename YEAR to period
        names[0] = 'period'
        assert len(data) == num_periods
        return names, data

    def load_csv_globals(self, input_path):
        with open(input_path, "rb") as f:
            lines = list(csv.reader(f))
        names, data = transpose_and_convert(lines)
        assert names[0].lower() == 'period'
        names[0] = 'period'
        return names, data

    def load_globals(self, input_path):
        print "*** globals ***"
        print "reading", input_path
        if input_path.endswith('.av'):
            return self.load_av_globals(input_path)
        elif input_path.endswith('.csv'):
            return self.load_csv_globals(input_path)

    def create_globals_table(self, h5file, globals):
        const_names, const_data = globals
        #FIXME: use declared types instead
        coltypes = [(name, float) if name != 'period' else ('period', int)
                    for name in const_names]
        dtype = np.dtype(coltypes)
        const_array = np.array(const_data, dtype=dtype)
        
        const_node = h5file.createGroup("/", "globals", "Globals")

        const_table = h5file.createTable(const_node, "periodic", dtype,
                                         title="Global time series")
        const_table.append(const_array)
        const_table.flush()

    def import_tsv(self, delimiter="\t", complib='zlib'):
        print "Importing in", self.h5
        globals = self.load_globals(self.globals_path)
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

        self.create_globals_table(h5file, globals)

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
                with open(file_path) as f:
                    numlines = sum(1 for line in f) - 1
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
                print " * storing (using %s compression)..." % complib
                filters = tables.Filters(complevel=5, complib=complib,
                                         fletcher32=True)
            else:
                filters = None
            table = h5file.createTable(entities_node, entity.name, main_dtype,
                                       title="%s table" % entity.name,
                                       filters=filters)
            table.append(array)
            table.flush()

            dtype = np.dtype(entity.per_period_fields)
            h5file.createTable(entities_node, entity.name + "_per_period",
                               dtype,
                               title="%s per period table" % entity.name,
                               expectedrows=12*50)

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
        
    def import_csv(self, delimiter=",", buffersize=10 * 2 ** 20): # 10 Mb
        print "Importing in", self.h5
        globals = self.load_globals(self.globals_path)
        h5file = tables.openFile(self.h5, mode="w", title="CSV import")
        self.create_globals_table(h5file, globals)
        ent_node = h5file.createGroup("/", "entities", "Entities")

        for ent_name, entity in entity_registry.iteritems():
            print "*** %s ***" % ent_name
            
            # per period table
            dtype = np.dtype(entity.per_period_fields)
            h5file.createTable(ent_node, ent_name + "_per_period", dtype,
                               title="%s per period table" % ent_name,
                               expectedrows=12*50)
            
            # main table
            oldnames = self.oldnames.get(ent_name, {})
            toinvert = self.toinvert.get(ent_name, [])
            filename = ent_name + ".csv"
            filepath = os.path.join(self.csv_path, filename)
            
            neededfields = [(name, type_) for name, type_ in entity.fields
                            if name not in set(entity.missing_fields)]
            neededfield_oldnames = [(oldnames.get(name, name), type_)
                                    for name, type_ in neededfields]
            
            dtype = np.dtype(neededfields)
            table = h5file.createTable(ent_node, ent_name, dtype,
                                       title="%s table" % ent_name)
            
            # count number of lines
            with open(filepath) as f:
                numlines = sum(1 for line in f) - 1
            
            datastream = import_csv(filepath, neededfield_oldnames, delimiter)

            # buffered load
            max_buffer_rows = buffersize / dtype.itemsize
            while numlines > 0:
                buffer_rows = min(numlines, max_buffer_rows)
                # ideally, we should preallocate an empty buffer and reuse it, 
                # but that does not seem to be supported by numpy
                buffer = np.fromiter(islice(datastream, max_buffer_rows),
                                     dtype=dtype, count=buffer_rows)
                for field in toinvert:
                    buffer[field] = ~buffer[field]
                table.append(buffer)
                table.flush()
                
                numlines -= buffer_rows

        h5file.close()
        

if __name__ == '__main__':
    import sys

    print 'Using Python %s' % sys.version
    args = sys.argv
    valid_cmds = ('export_tsv', 'export_csv', 'import_tsv', 'import_csv')
    if len(args) < 4:
        print """
Usage: %s action simulation_file import_export_file
  action can be any of %s
""" % (args[0], ', '.join(repr(cmd) for cmd in valid_cmds))
        sys.exit()
    cmd = args[1]
    
    print "using simulation file: '%s' / '%s'" % (args[2], args[3])
    data = ImportExportData(args[2], args[3])

    assert cmd in valid_cmds, "Unknown command '%s'" % cmd
    timed(getattr(data, cmd))
