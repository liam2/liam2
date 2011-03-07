import tables
import numpy as np

from expr import type_to_idx, idx_to_type
from properties import missing_values

def assertValidFields(s_fields, table, allowed_missing=None):
    # extract types from field description and normalise to python types
    t_fields = [(k, idx_to_type[type_to_idx[v[0].type]])
                for k, v in table.dtype.fields.iteritems()]

    # check that all required fields are present
    s_names = set(name for name, _ in s_fields)
    t_names = set(name for name, _ in t_fields)
    allowed_missing = set(allowed_missing) if allowed_missing is not None \
                                           else set() 
    missing = s_names - t_names - allowed_missing 
    if missing:
        raise Exception("Missing field(s) in hdf5 input file: %s"
                        % ', '.join(missing))

    # check that types match
    common_t_fields = [(name, type_)
                       for name, type_ in t_fields if name in s_names]
    common_s_fields = [(name, type_)
                       for name, type_ in s_fields if name in t_names] 
    bad_fields = []
    for (name1, t1), (name2, t2) in zip(sorted(common_s_fields),
                                        sorted(common_t_fields)):
        assert name1 == name2
        if t1 != t2:
            bad_fields.append((name1, t2.__name__, t1.__name__))
    if bad_fields:
        bad_fields_str = "\n".join(" - %s: %s instead of %s" % f for f in bad_fields)
        raise Exception("Field types in hdf5 input file differ from those "
                        "defined in the simulation:\n%s" % bad_fields_str)

def add_and_drop_fields(array, output_fields):
    output_dtype = np.dtype(output_fields)
    output_names = set(output_dtype.names)
    input_names = set(array.dtype.names)
    common_fields = output_names & input_names 
    missing_fields = output_names - input_names 
    output_array = np.empty(len(array), dtype=output_dtype)
    for fname in common_fields:
        output_array[fname] = array[fname]
    for fname in missing_fields:
        ftype = idx_to_type[type_to_idx[output_dtype.fields[fname][0].type]]
        output_array[fname] = missing_values[ftype]
    return output_array

def copyTable(input_table, output_file, output_node, output_fields,
              chunksize=10000, **kwargs):
    complete_kwargs = {'title': input_table._v_title,
                       'filters': input_table.filters}
    complete_kwargs.update(kwargs)
    if output_fields is None:
        output_dtype = input_table.dtype
    else:
        output_dtype = np.dtype(output_fields)
    output_table = output_file.createTable(output_node, input_table.name, 
                                           output_dtype, **complete_kwargs)
    rowsleft = len(input_table)
    if chunksize is None:
        chunksize = rowsleft
        returndata = True
    else:
        returndata = False 

    start = 0
    num_chunks, remainder = divmod(rowsleft, chunksize)
    if remainder > 0:
        num_chunks += 1
    
    for _ in range(num_chunks):
        stop = start + min(chunksize, rowsleft)
        data = input_table.read(start, stop)
        if output_fields is not None:
            data = add_and_drop_fields(data, output_fields)
        output_table.append(data)
        output_table.flush()
        rowsleft -= chunksize
        start = stop

    return data if returndata else None


class H5Data(object):
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        
    def run(self, entities, start_period):
        last_period = start_period - 1
        print "reading data from %s and storing period %d in memory..." % \
              (self.input_path, last_period)

        input_file = tables.openFile(self.input_path, mode="r")
        output_file = tables.openFile(self.output_path, mode="w")

        input_root = input_file.root

        output_globals = output_file.createGroup("/", "globals", 
                                                   "Globals")
        # load globals in memory
        # TODO: use the built-in Table.copy(), which should work fine starting
        # from PyTables 2.2.1, or use copy entire nodes, see below.  
        periodic_globals = copyTable(input_root.globals.periodic,
                                     output_file, output_globals, None,
                                     chunksize=None)

        input_entities = input_root.entities
        output_entities = output_file.createGroup("/", "entities", "Entities")
    
        # copy nodes as a batch
#        globals_node = input_file.copyNode("/globals", output_file.root)
#        entities_node = input_file.copyNode("/entities", output_file.root)
        
        res = {}
        for ent_name, entity in entities.iteritems():
            print ent_name, "..."
            
            # main table
            table = getattr(input_entities, ent_name)

            assertValidFields(entity.fields, table, entity.missing_fields)

            # build index
            print "building table index...",
            table_index = {}
            current_period = None
            start_row = None
            for idx, row in enumerate(table):
                period = row['period']
                if period != current_period:
                    # 0 > None is True
                    assert period > current_period, "data is not time-ordered"
                    if start_row is not None:
                        table_index[current_period] = start_row, idx
                    start_row = idx
                    current_period = period
            table_index[current_period] = (start_row, len(table))
            entity.period_rows = table_index
            entity.base_period = min(table_index.iterkeys())
            print "done."

            # check that there is not too much data in input file
#            max_period = max(table_index.iterkeys())
#            assert max_period < start_period, \
#                   "invalid period(s) found in input file: all data must be " \
#                   "for past periods (period < start_period)"
            
            # copy stuff
            print "copying tables & loading last period...",
            copyTable(table, output_file, output_entities, entity.fields)

            # load last period in memory
            if last_period in table_index:
                start, stop = table_index[last_period]
                array = table.read(start=start, stop=stop)
                array = add_and_drop_fields(array, entity.fields)
            else:
                dtype = np.dtype(entity.fields)
                array = np.empty(0, dtype=dtype)
            
            # per period
            per_period_table = getattr(input_entities, ent_name + "_per_period")
            assertValidFields(entity.per_period_fields,
                              per_period_table, entity.pp_missing_fields)
            
            copyTable(per_period_table, output_file, output_entities,
                      entity.per_period_fields)
            per_period_array = per_period_table.readWhere("period==%d" 
                                                          % last_period)
            per_period_array = add_and_drop_fields(per_period_array,
                                                   entity.per_period_fields)
            if not len(per_period_array):
                #TODO: use missing values instead
                per_period_array = np.zeros(1, dtype=per_period_array.dtype)
                per_period_array['period'] = last_period

            res[ent_name] = array, per_period_array
            print "done."
            
        input_file.close()
        output_file.close()
        return periodic_globals, res

    
class Void(object):
    def __init__(self, output_path):
        self.output_path = output_path
        
    def run(self, entities, start_period):
        output_file = tables.openFile(self.output_path, mode="w")

        output_globals = output_file.createGroup("/", "globals", "Globals")
        dtype = np.dtype([('period', int)])
        periodic_globals = np.empty(0, dtype=dtype)
        output_file.createTable(output_globals, 'periodic', dtype, title='')

        output_entities = output_file.createGroup("/", "entities", "Entities")

        res = {}
        for ent_name, entity in entities.iteritems():
            dtype = np.dtype(entity.fields)
            array = np.empty(0, dtype=dtype)
            output_file.createTable(output_entities, entity.name, dtype,
                                    title="%s table" % entity.name)

            dtype = np.dtype(entity.per_period_fields)
            per_period_array = np.empty(0, dtype=dtype)
            output_file.createTable(output_entities,
                                    entity.name + "_per_period",
                                    dtype,
                                    title="%s per period table" % entity.name)
            res[ent_name] = array, per_period_array
        
        output_file.close()
        return periodic_globals, res

