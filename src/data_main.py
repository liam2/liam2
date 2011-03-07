import os
import csv
from itertools import izip

import numpy as np
import tables
import yaml

from entities import entity_registry
from utils import timed
from properties import missing_values

from simulation import Simulation
        


def dicttotuple(record, fields):
    return tuple([record.get(fname, default) for fname, default in fields])

def index_first(header, possible_values):
    values_present = set(header) & set(possible_values)
    assert len(values_present) == 1, "%s not found in header: %s" % (
                                     possible_values, header)
    colname = values_present.pop()
    return header.index(colname)


def to_int(v):
    return int(float(v)) if v else 0

def to_float(v):
    return float(v) if v else 0.0

def to_bool(v):
    return v.lower() in ('1', 'true')

converters = {bool: to_bool,
            int: to_int,
            float: to_float}

def import_csv_3col(fpath, vname, vtype, delimiter=","):
    '''imports one Xsv file which use a three columns format: 
       id, time, value (not necessarily in that order)
    ''' 
    print "reading", fpath, "(%s)" % vname

    func = converters[vtype]

    with open(fpath, "rb") as f:
        dataReader = csv.reader(f, delimiter=delimiter)
        
        header = dataReader.next()
        id_pos = index_first(header, ['id', 'idnum', 'oid'])
        time_pos = index_first(header, ['time', 'year', 'period'])
        value_pos = index_first(header, ['value', 'did', vname])
        for row in dataReader:
            yield (int(row[id_pos]), int(row[time_pos]), func(row[value_pos]))


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
    
def load_objtype(fpath):
    with open(fpath, "rb") as f:
        dataReader = csv.reader(f, delimiter='\t')
        header = dataReader.next()
        v = header[0]
        try:
            int(v)
            raised = False
        except ValueError:
            raised = True
        assert raised, "first line must be the column name"
            
        lines = list(dataReader)
    return np.array([int(line[0]) for line in lines])

def transpose_table(data):
    numrows = len(data)
    numcols = len(data[0])
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

    def create_tables(self, h5file, globals):
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

        ent_node = h5file.createGroup("/", "entities", "Entities")
#        typemap = {bool: np.int8}
        typemap = {}
        for entity in entity_registry.itervalues():
#            fields = [(name, typemap.get(type_, type_))
#                      for name, type_ in entity.fields]
            neededfields = [(name, type_) for name, type_ in entity.fields
                            if name not in set(entity.missing_fields)]
            dtype = np.dtype(neededfields)
            table = h5file.createTable(ent_node, entity.name, dtype,
                                       title="%s table" % entity.name)
            table.append(entity.array)
            table.flush()

            dtype = np.dtype(entity.per_period_fields)
            per_period_table = \
                h5file.createTable(ent_node, entity.name + "_per_period", dtype,
                                   title="%s per period table" % entity.name,
                                   expectedrows=12*50)

#            per_period_table.append(entity.per_period_array)              
#            per_period_table.flush()

    def import_tsv(self, delimiter="\t"):
        print "Importing in", self.h5
        globals = self.load_globals(self.globals_path)
        h5file = tables.openFile(self.h5, mode="w", title="CSV import")
        missing_files, loaded_files = [], []
        for ent_name, entity in entity_registry.iteritems():
            print "*** %s ***" % ent_name 
            oldnames = self.oldnames.get(ent_name, {})
            toinvert = self.toinvert.get(ent_name, [])

            neededfields = [(name, type_) for name, type_ in entity.fields
                            if name not in set(entity.missing_fields)]

            many2one_links = [v for v in entity.links.itervalues()
                              if v._link_type == 'many2one']
            fields_toskip = set([link._link_field for link in many2one_links])
            fields_toskip.update(["id", "period"])
            
            toload = []
            # list "fields" to be loaded
            for vname, vtype in neededfields:
                if vname not in fields_toskip:
                    oldname = oldnames.get(vname, vname)
                    filename = "%s_%s.txt" % (ent_name[0], oldname)
                    toload.append((filename, vname, oldname, vtype))
            # list "links" to be loaded
            for link in many2one_links:
                vname = link._link_field
                oldname = oldnames.get(vname, vname)
                filename = "link_%s.txt" % link._name
                toload.append((filename, vname, oldname, int))

            # load them into memory
            records = {}
            for filename, vname, oldname, vtype in toload:
                file_path = os.path.join(self.csv_path, filename)
                if os.path.exists(file_path):
                    for id, period, value in import_csv_3col(file_path, 
                                                             oldname, vtype,
                                                             delimiter):
                        assert period < self.simulation.start_period, \
                            "%s has observations for period %d beyond simulation start period %d"  %(vname, period, self.simulation.start_period)
                            
                        k = (period, id)
                        if k in records:
                            records[k][vname] = value
                        else:
                            records[k] = {'id': id, 'period': period,
                                          vname: value}
                    loaded_files.append(filename)
                else:
                    missing_files.append(filename)

            print "sorting..."
            sorted_keys = sorted(records.iterkeys())
            
            # transform our dictionaries into arrays
            print "indexing..."
            periods_for_id = {}
            for period, id in sorted_keys:
                periods_for_id.setdefault(id, []).append(period)
            print "to list..."
            fdescs = [(fname, missing_values[ftype]) for fname, ftype in neededfields]
            
            needed_fnames = set(fname for fname, ftype in neededfields)
            # by completing the data with previous periods, where available
            l_items = []
            for period, id in sorted_keys:
                record = records[(period, id)]
                missing_fields = needed_fnames - set(record.keys())
                if missing_fields:
                    id_periods = periods_for_id[id]
                    period_idx = id_periods.index(period)
                    if period_idx > 0:
                        # go back as far as necessary to find values for all
                        # the missing fields of that record
                        previous_periods = id_periods[period_idx - 1::-1]
                        for previous_period in previous_periods:
                            previous_record = records[(previous_period, id)]
                            for fname in tuple(missing_fields):
                                value = previous_record.get(fname)
                                if value is not None:
                                    record[fname] = value
                                    missing_fields.remove(fname)
                            if not missing_fields:
                                break
                    # if there are still missing values, use the defaults
                    if missing_fields:
                        for fname, default in fdescs:
                            if fname in missing_fields:
                                record[fname] = default                     
                l_items.append(tuple([record[fname] for fname, _ in neededfields]))

            # version with only missing values    
#            l_items = [dicttotuple(records[k], fdescs) for k in sorted_keys]

            print "to array..."
            array = np.array(l_items, dtype=np.dtype(neededfields))
            for field in toinvert:
                array[field] = ~array[field]
            entity.array = array

            filename = "objtype_%s.txt" % ent_name
            loaded_files.append(filename)
            print "generating id_to_rownum (loading %s)..." % filename
            file_path = os.path.join(self.csv_path, filename)
            ids = load_objtype(file_path)
            max_id = np.max(ids)
            id_to_rownum = np.empty(max_id + 1, dtype=int)
            id_to_rownum[:] = -1
            for idx, id in enumerate(ids):
                id_to_rownum[id] = idx 
            entity.id_to_rownum = id_to_rownum
#            dtype = np.dtype(entity.per_period_fields)
#            entity.per_period_array = np.zeros(1, dtype=dtype)

        # this is done so that id_to_rownum arrays are smaller
        print "shifting ids and links..."
        for ent_name, entity in entity_registry.iteritems():
            to_shift = [(l._link_field, l._target_entity) 
                        for l in entity.links.values()
                        if l._link_type == 'many2one']
            to_shift.append(('id', ent_name))
            print " * shifting %s fields:" % ent_name
            for field, target_name in to_shift:
                print "   -", field
                target = entity_registry[target_name]
                array = entity.array[field]
                missing_int = missing_values[int]
                id_to_rownum = target.id_to_rownum
                entity.array[field] = np.where(array == missing_int,
                                               missing_int, 
                                               id_to_rownum[array])

        print "storing..."
        self.create_tables(h5file, globals)
        h5file.close()
        
        if missing_files:
            raise Exception("missing files:\n - %s" 
                            % "\n - ".join(missing_files))

#        present_files = os.listdir(self.csv_path)
#        extra_files = sorted(list(set(present_files) - set(loaded_files)))
#        print "extra files", extra_files
        
    def import_csv(self, delimiter=","):
        print "Importing in", self.h5
        globals = self.load_globals(self.globals_path)
        h5file = tables.openFile(self.h5, mode="w", title="CSV import")

        for ent_name, entity in entity_registry.iteritems():
            print "*** %s ***" % ent_name
            oldnames = self.oldnames.get(ent_name, {})
            toinvert = self.toinvert.get(ent_name, [])
            filename = ent_name + ".csv"
            filepath = os.path.join(self.csv_path, filename)
            neededfields = [(name, type_) for name, type_ in entity.fields
                            if name not in set(entity.missing_fields)]
            neededfield_oldnames = [(oldnames.get(name, name), type_)
                                    for name, type_ in neededfields]
            
            data = import_csv(filepath, neededfield_oldnames, delimiter)
            array = np.array(list(data), dtype=np.dtype(neededfields))
            for field in toinvert:
                array[field] = ~array[field]
            entity.array = array
            
        print "storing..."
        self.create_tables(h5file, globals)
        h5file.close()
        

if __name__ == '__main__':
    import sys

    print 'Using Python %s' % sys.version
    args = sys.argv
    if len(args) < 4:
        print """
Usage: %s action simulation_file import_export_file
  action can be any of 'export_tsv', 'export_csv', 'import_tsv', 'import_csv'
""" % args[0]
        sys.exit()
    cmd = args[1]
    
    print "using simulation file: '%s' / '%s'" % (args[2], args[3])
    data = ImportExportData(args[2], args[3])

    assert cmd in ('export_tsv', 'export_csv', 'import_tsv', 'import_csv'), \
           "Unknown command '%s'" % cmd
    timed(getattr(data, cmd))
