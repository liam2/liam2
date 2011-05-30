import time

import tables
import numpy as np

from expr import normalize_type, get_missing_value, get_missing_record
from utils import loop_wh_progress, time2str, safe_put

def table_size(table):
    return (len(table) * table.dtype.itemsize) / 1024.0 / 1024.0

def get_fields(array):
    dtype = array.dtype
    field_types = dtype.fields
    return [(name, normalize_type(field_types[name][0].type))
            for name in dtype.names]

def assertValidFields(s_fields, array, allowed_missing=None):
    # extract types from field description and normalise to python types
    t_fields = get_fields(array)

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

def mergeSubsetInArray(output, id_to_rownum, subset, first=False):
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
        #TODO: this is a gross approximation, more research is needed to get
        # a better threshold. It might also depend on "first".
        if len(names_to_copy) > len(output_names) / 2:
            if first:
                subset_all_cols = np.empty(len(subset), dtype=output.dtype)
                for fname in set(output_names) - set(subset_names):
                    subset_all_cols[fname] = \
                        get_missing_value(subset_all_cols[fname])
            else:
                subset_all_cols = output[rownums]
                # Note that all rows which correspond to rownums == -1 have wrong
                # values (they have the value of the last row) but it is not 
                # necessary to correct them since they will not be copied back
                # into output_array.
                # np.putmask(subset_all_cols, rownums == -1, missing_row)
            for fname in names_to_copy:
                subset_all_cols[fname] = subset[fname]
            safe_put(output, rownums, subset_all_cols)
        else:
            for fname in names_to_copy:
                safe_put(output[fname], rownums, subset[fname])
        return output
    
def mergeArrays(array1, array2, result_fields='union'):
    fields1 = get_fields(array1)
    fields2 = get_fields(array2)
    #TODO: check that common fields have the same type
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
    for rownum, id in enumerate(all_ids):
        id_to_rownum[id] = rownum

    # 1) create resulting array
    ids1_complete = len(ids1) == len(all_ids)
    ids2_complete = len(ids2) == len(all_ids)
    output_is_arr1 = array1.dtype == output_dtype and ids1_complete
    output_is_arr2 = array2.dtype == output_dtype and ids2_complete
    arr1_complete = set(fields1) >= set(output_fields) and ids1_complete
    arr2_complete = set(fields2) >= set(output_fields) and ids2_complete
    if output_is_arr1 or output_is_arr2:
        # In this case, output_array will be one of the input arrays, so
        # we don't need to allocate/initialise it
        pass
    elif arr1_complete or arr2_complete:
        output_array = np.empty(len(all_ids), dtype=output_dtype)
    else:
        output_array = np.empty(len(all_ids), dtype=output_dtype)
        output_array[:] = get_missing_record(output_array)
    
    # 2) copy data from array1
    if not arr2_complete:
        if output_is_arr1:
            output_array = array1
        else:
            output_array = mergeSubsetInArray(output_array, id_to_rownum, 
                                              array1, first=True)

    # 3) copy data from array2
    if output_is_arr2:
        output_array = array2
    else:
        output_array = mergeSubsetInArray(output_array, id_to_rownum, array2)
    
    return output_array, id_to_rownum


def appendTable(input_table, output_table, chunksize=10000, condition=None,
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
            
    def copyChunk(chunk_idx, chunk_num):
        start = chunk_num * chunksize
        stop = min(start + chunksize, numrows)
        if condition is not None:
            input_data = input_table.readWhere(condition,
                                               start=start, stop=stop)
        else:
            input_data = input_table.read(start, stop)

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
        loop_wh_progress(copyChunk, range(num_chunks))
    else:
        for chunk in range(num_chunks):
            copyChunk(chunk, chunk)

    return output_table

def copyTable(input_table, output_file, output_node, output_fields=None,
              chunksize=10000, condition=None, stop=None, show_progress=False, 
              **kwargs):
    complete_kwargs = {'title': input_table._v_title,
                       'filters': input_table.filters}
    complete_kwargs.update(kwargs)
    if output_fields is None:
        output_dtype = input_table.dtype
    else:
        output_dtype = np.dtype(output_fields)
    output_table = output_file.createTable(output_node, input_table.name, 
                                           output_dtype, **complete_kwargs)
    return appendTable(input_table, output_table, chunksize, condition, 
                       stop=stop, show_progress=show_progress)

#XXX: should I make a generic n-way array merge out of this?
# this is a special case though because: 
# 1) all arrays have the same columns
# 2) we have id_to_rownum already computed for each array 
def buildArrayForPeriod(input_table, output_fields, input_rows, input_index,
                        max_id_per_period, target_period):
    periods_before = [p for p in input_rows.iterkeys() if p <= target_period]
    periods_before.sort()

    # computing is present
    max_id = max_id_per_period[target_period]
    is_present = np.zeros(max_id + 1, dtype=bool)
    for period in periods_before:
        period_id_to_rownum = input_index[period]
        present_in_period = period_id_to_rownum != -1
        present_in_period.resize(max_id + 1)
        is_present |= present_in_period

    if np.array_equal(present_in_period, is_present):
        start, stop = input_rows[target_period]
        input_array = input_table.read(start=start, stop=stop)
        return (add_and_drop_fields(input_array, output_fields), 
                period_id_to_rownum)
         
    # building id_to_rownum for the target period
    id_to_rownum = np.empty(max_id + 1, dtype=int)
    id_to_rownum.fill(-1)
    rownum = 0
    for id, present in enumerate(is_present):
        if present:
            id_to_rownum[id] = rownum
            rownum += 1
            
#    all_ids = is_present.nonzero()[0] 

    # computing source row (period) for each destination row    
    output_array_source_rows = np.empty(rownum, dtype=int)
    output_array_source_rows.fill(-1)
    for period in periods_before[::-1]:
#        missing_rows = output_array_source_rows == -1
#        if not np.any(missing_rows):
#            break
#        start, stop = input_rows[period]
#        missing_ids = all_ids[missing_rows]
#        input_id_to_rownum = input_index[period]
#        input_rownums_for_missing = input_id_to_rownum[missing_ids] + start  
#        output_array_source_rows[missing_rows] = input_rownums_for_missing 
        
        start, stop = input_rows[period]
        input_id_to_rownum = input_index[period]
        input_ids = (input_id_to_rownum != -1).nonzero()[0]
        input_rownums = np.arange(start, stop)
        output_rownums = id_to_rownum[input_ids]
        source_rows = output_array_source_rows[output_rownums]
        np.putmask(source_rows, source_rows == -1, input_rownums)
        safe_put(output_array_source_rows, output_rownums, source_rows)

        if np.all(output_array_source_rows != -1):
            break

#        start, stop = input_rows[period]
#        for id, input_rownum in enumerate(input_index[period]):
#            if input_rownum != -1:
#                output_rownum = id_to_rownum[id]
#                if output_rownum != -1:
#                    if output_array_source_rows[output_rownum] == -1:
#                        output_array_source_rows[output_rownum] = \
#                            start + input_rownum
                            
#        if np.all(output_array_source_rows != -1):
#            break
        
    # reading data 
    output_array = input_table.readCoordinates(output_array_source_rows)
    output_array = add_and_drop_fields(output_array, output_fields)

    return output_array, id_to_rownum
    

class H5Data(object):
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        
    def run(self, entities, start_period):
        last_period = start_period - 1
        print "reading data from %s ..." % self.input_path

        input_file = tables.openFile(self.input_path, mode="r")
        output_file = tables.openFile(self.output_path, mode="w")

        input_root = input_file.root

        output_globals = output_file.createGroup("/", "globals", "Globals")
        # load globals in memory
        copyTable(input_root.globals.periodic, output_file, output_globals)
        periodic_globals = input_root.globals.periodic.read()

        input_entities = input_root.entities
        output_entities = output_file.createGroup("/", "entities", "Entities")
    
        for ent_name, entity in entities.iteritems():
            print ent_name, "..."
            
            # main table
            table = getattr(input_entities, ent_name)

            assertValidFields(entity.fields, table, entity.missing_fields)

            # build indexes
            print " * indexing table...",
            start_time = time.time()
            table_index = {}
            current_period = None
            start_row = None
            max_id_per_period = {}
            max_id_so_far = -1
            temp_id_to_rownum = []
            for idx, row in enumerate(table):
                period, id = row['period'], row['id']
                if period != current_period:
                    # 0 > None is True
                    assert period > current_period, "data is not time-ordered"
                    if start_row is not None:
                        table_index[current_period] = start_row, idx
                        # assumes the data is sorted on period then id
                        max_id_per_period[current_period] = max_id_so_far
                        id_to_rownum = np.array(temp_id_to_rownum)
                        entity.input_index[current_period] = id_to_rownum
                        temp_id_to_rownum = [-1] * (max_id_so_far + 1)
                    start_row = idx
                    current_period = period
                if id > max_id_so_far:
                    extra = [-1] * (id - max_id_so_far)
                    temp_id_to_rownum.extend(extra)
                temp_id_to_rownum[id] = idx - start_row
                max_id_so_far = max(max_id_so_far, id)
            table_index[current_period] = (start_row, len(table))
            max_id_per_period[current_period] = max_id_so_far
            entity.input_index[current_period] = np.array(temp_id_to_rownum)

            entity.input_rows = table_index
            entity.base_period = min(table_index.keys())
            print "done (%s elapsed)." % time2str(time.time() - start_time)

            print " * copying table..."
            start_time = time.time()
            input_rows = entity.input_rows
            output_rows = dict((p, rows) for p, rows in input_rows.iteritems()
                               if p < last_period)
            if output_rows:
                _, stoprow = input_rows[max(output_rows.iterkeys())]
            else:
                stoprow = 0
            
            copyTable(table, output_file, output_entities, 
#                      entity.fields, stop=0, show_progress=True)
                      entity.fields, stop=stoprow, show_progress=True)
            entity.output_rows = output_rows
            print "done (%s elapsed)." % time2str(time.time() - start_time)
            
            print " * building array for first simulated period...",
            start_time = time.time()
            
            array, id_to_rownum = \
                buildArrayForPeriod(table, entity.fields, entity.input_rows, 
                                    entity.input_index, max_id_per_period,
                                    last_period)
            entity.id_to_rownum = id_to_rownum
            entity.array = array
            print "done (%s elapsed)." % time2str(time.time() - start_time)

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
            
            entity.per_period_array = per_period_array
            
        input_file.close()
        output_file.close()
        return periodic_globals

    
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

        for ent_name, entity in entities.iteritems():
            dtype = np.dtype(entity.fields)
            entity.array = np.empty(0, dtype=dtype)
            output_file.createTable(output_entities, entity.name, dtype,
                                    title="%s table" % entity.name)

            dtype = np.dtype(entity.per_period_fields)
            entity.per_period_array = np.empty(0, dtype=dtype)
            output_file.createTable(output_entities,
                                    entity.name + "_per_period",
                                    dtype,
                                    title="%s per period table" % entity.name)
        
        output_file.close()
        return periodic_globals

