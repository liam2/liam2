import time

import tables
import numpy as np

from expr import normalize_type, get_missing_value, get_missing_record, \
                 missing_values
from utils import loop_wh_progress, time2str, safe_put


def table_size(table):
    return (len(table) * table.dtype.itemsize) / 1024.0 / 1024.0


def get_fields(array):
    dtype = array.dtype
    field_types = dtype.fields
    return [(name, normalize_type(field_types[name][0].type))
            for name in dtype.names]


def assertValidFields(array, required_fields):
    # extract types from field description and normalise to python types
    actual_fields = get_fields(array)

    # check that all required fields are present
    required_names = set(name for name, _ in required_fields)
    actual_names = set(name for name, _ in actual_fields)
    missing = sorted(required_names - actual_names)
    if missing:
        raise Exception("Missing field(s) in input data: %s"
                        % ', '.join(missing))

    # check that types match
    sorted_subset = sorted((name, type_) for name, type_ in actual_fields
                           if name in required_names)
    sorted_req_fields = sorted(required_fields)
    bad_fields = []
    for (name1, t1), (name2, t2) in zip(sorted_subset, sorted_req_fields):
        assert name1 == name2, "%s != %s" % (name1, name2)
        if t1 != t2:
            bad_fields.append((name1, t2.__name__, t1.__name__))
    if bad_fields:
        bad_fields_str = "\n".join(" - %s: %s instead of %s" % f
                                   for f in bad_fields)
        raise Exception("Field types in input data differ from those "
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
        output_array = array1
    elif arr1_complete or arr2_complete:
        output_array = np.empty(len(all_ids), dtype=output_dtype)
    else:
        output_array = np.empty(len(all_ids), dtype=output_dtype)
        output_array[:] = get_missing_record(output_array)

    # 2) copy data from array1
    if not arr2_complete:
        output_array = mergeSubsetInArray(output_array, id_to_rownum,
                                          array1, first=True)

    # 3) copy data from array2
    if not output_is_arr2:
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
                      }
#                       'filters': input_table.filters}
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
                        start_period):
    periods_before = [p for p in input_rows.iterkeys() if p <= start_period]
    if not periods_before:
        id_to_rownum = np.empty(0, dtype=int)
        output_array = np.empty(0, np.dtype(output_fields))
        return output_array, id_to_rownum

    periods_before.sort()
    # take the last period which we have data for
    target_period = periods_before[-1]

    # computing is present
    max_id = len(input_index[target_period]) - 1
    is_present = np.zeros(max_id + 1, dtype=bool)
    for period in periods_before:
        period_id_to_rownum = input_index[period]
        present_in_period = period_id_to_rownum != -1
        present_in_period.resize(max_id + 1)
        is_present |= present_in_period

    # if all individuals are present in the target period, we are done already!
    if np.array_equal(present_in_period, is_present):
        start, stop = input_rows[target_period]
        input_array = input_table.read(start=start, stop=stop)
        return (add_and_drop_fields(input_array, output_fields),
                period_id_to_rownum)

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
    output_array = input_table.readCoordinates(output_array_source_rows)
    output_array = add_and_drop_fields(output_array, output_fields)

    return output_array, id_to_rownum


def index_table(table):
    '''
    table is an iterable of rows, each row is a mapping (name -> value).
    Rows must contain at least 'period' and 'id' columns and must be sorted
    by period.
    '''
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


class ListOfMaps(list):
    '''helper class to make a list of dict behave "like a context"
    (ie the get method that is used in evaluate to see if the variable
    is already in the context)
    maybe that's not what I need: maybe I should have a "context" object
    between XXXTable and evaluate?
    '''
    def __getitem__(self, key):
        if isinstance(key, basestring):
            return [row[key] for row in self]
        else:
            return list.__getitem__(self, key)

    def __contains__(self, key):
        if isinstance(key, basestring):
            return key in self[0]
        else:
            return list.__getitem__(self, key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def index_table_light(table):
    '''
    table is an iterable of rows, each row is a mapping (name -> value)
    Rows must contain at least a 'period' column and must be sorted by period.
    '''
    rows_per_period = {}
    current_period = None
    start_row = None
    for idx, row in enumerate(table):
        period = row['period']
        if period != current_period:
            # 0 > None is True
            if period < current_period:
                raise Exception("data is not time-ordered")
            if start_row is not None:
                rows_per_period[current_period] = (start_row, idx)
            start_row = idx
            current_period = period
    if current_period is not None:
        rows_per_period[current_period] = (start_row, len(table))
    return rows_per_period


class PeriodTable(object):
    def __init__(self, data, period):
        self.data = data
        self.period = period

    def __getitem__(self, fieldname):
        return self.data.read(self.period, field=fieldname)

    def __contains__(self, key):
        return key in self.data

    @property
    def id_to_rownum(self):
        return self.data.id2rownum_per_period[self.period]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __len__(self):
        return len(self.data)


class FilteredTable(object):
    def __init__(self, period_table, ids, filter_missing_ids=True):
        self.period_table = period_table
        self.ids = ids
        self.filter_missing_ids = filter_missing_ids
        self.rows = period_table.id_to_rownum[ids]

    def __getattr__(self, key):
        return getattr(self.period_table, key)

    def __getitem__(self, fieldname):
        try:
            fid = self.filter_missing_ids
            return self.period_table.data.read(self.period_table.period,
                                               self.ids,
                                               field=fieldname,
                                               filter_missing_ids=fid)
        except ValueError:
            raise KeyError

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        return key in self.period_table


class IndexedTable(object):
    def __init__(self, table):
        self.table = table
        self.period_index, self.id2rownum_per_period = index_table(table)

    def __getitem__(self, period):
        if isinstance(period, basestring):
            raise Exception("yargl")
        return PeriodTable(self, period)

    def __contains__(self, key):
        return key in self.table.dtype.fields

    def get(self, key, default=None):
        try:
            return self.table[key]
        except KeyError:
            return default

    def __len__(self):
        return len(self.table)

#XXX: might be cleaner/more logical to move this to PeriodTable instead
    # In the future, we will probably want a more flexible interface, but
    # it will need a lot more machinery (query language and so on) so let's
    # stick with something simple for now.
    def read(self, period, ids=None, field=None, filter_missing_ids=False):
        if period not in self.period_index:
            raise Exception('no data for period %d' % period)
        start, stop = self.period_index[period]
        if ids is None:
            # this is probably faster, but it is not compatible with
            # in-memory arrays
#            return self.table.read(start=start, stop=stop, field=field)
            return self.table[start:stop][field]
        else:
            id_to_rownum = self.id2rownum_per_period[period]
            missing_int = missing_values[int]
            #XXX: this might be inefficient because we'll recompute
            #valid_ids & target_rows for each field in the expression
            target_rows = id_to_rownum[ids]
            if filter_missing_ids:
                valid_ids = (ids != missing_int) & (target_rows != missing_int)
            else:
                assert np.all(ids != missing_int)
                valid_ids = target_rows != missing_int
            target_rows += start

            # this is probably faster, but it is not compatible with
            # in-memory arrays
#            target_values = self.table.readRows(target_rows, field=field)
            target_values = self.table[target_rows][field]
            missing_value = get_missing_value(target_values)
            return np.where(valid_ids, target_values, missing_value)

    @property
    def base_period(self):
        return min(self.period_index.keys())


class DictNodeWrapper(object):
    def __init__(self, node):
        self.node = node

    def __getitem__(self, key):
        child = self.node._f_getChild(key)
        if isinstance(child, tables.Leaf):
            return child
        else:
            return DictNodeWrapper(child)


def validate_dataset(dataset, dataset_def):
    '''
    check that a data set contains at least all the tables and groups defined
    in dataset_def, that the tables contain at least the fields described
    for each table and those fields are of the correct type.
    '''
    assert isinstance(dataset_def, dict)
    for k, node_def in dataset_def.iteritems():
        try:
            node_data = dataset[k]
        except KeyError:
            raise Exception('Missing node in input dataset', k)
        if isinstance(node_def, dict):
            # we have a group structure
            validate_dataset(node_data, node_def)
        else:
            # we have a list of fields
            assert isinstance(node_def, list), str(node_def)
            assertValidFields(node_data, node_def)


#XXX: inherit from dict instead of having a .data dict attribute?
class DataSet(object):
    pass


#TODO: indexed vs full_load should be transparent to the outside world
#XXX: would it be a good idea to make the decision of indexing or full load
# automatic (if a table is smaller than X)?
def index_tables(input_dataset, dataset_def):
    '''
    index or load all tables present in dataset_def which should have the
    following format: dict for group nodes, boolean for table nodes. eg
    {'group_name':
        {'subgroup':
            ...
            {'tablename': full_load}
    where full_load determines whether the table is loaded entirely in memory
    or is only indexed.
    '''
    assert isinstance(dataset_def, dict)
    output_dataset = {}
    for node_name, node_def in dataset_def.iteritems():
        node_data = input_dataset[node_name]
        if isinstance(node_def, dict):
            # we have a group structure
            output_dataset[node_name] = index_tables(node_data, node_def)
        else:
            # we have a list of fields
            assert isinstance(node_def, bool)
            full_load = node_def
            if full_load:
                output_dataset[node_name] = node_data.read()
            else:
                print " * indexing table:", node_name, "...",
                start_time = time.time()
                output_dataset[node_name] = IndexedTable(node_data)
                print "done (%s elapsed)." % time2str(time.time() - start_time)

    return output_dataset


class DataSource(object):
    pass


def create_dataset_def(entities, globals_fields):
    entities_def = {}
    for name, entity in entities.iteritems():
        allowed_missing = set(entity.missing_fields)
        # compute required fields
        entities_def[name] = [(fname, ftype) for fname, ftype in entity.fields
                           if fname not in allowed_missing]
    return {'globals': {'periodic': globals_fields},
            'entities': entities_def}


def create_dataset_index_def(entities):
    # periodic globals shouldn't be indexed
    return {'globals': {'periodic': True},
            'entities': dict((name, False) for name in entities)}


# A data source is not necessarily read-only, but should be connected to
# only one file, so in our case we should have one instance for input and the
# other (used both for read and write) for the output.
class H5Data(DataSource):
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def load(self, globals_fields, entities):
        print "reading data from %s ..." % self.output_path
        h5file = tables.openFile(self.output_path, mode="r")
        dataset_def = create_dataset_def(entities, globals_fields)
        raw_dataset = DictNodeWrapper(h5file.root)
        validate_dataset(raw_dataset, dataset_def)

        dataset_def = create_dataset_index_def(entities)
        dataset = index_tables(raw_dataset, dataset_def)

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

        return h5file, None, dataset['globals']['periodic']

    def run(self, globals_fields, entities, start_period):
        print "reading data from %s ..." % self.input_path
        input_file = tables.openFile(self.input_path, mode="r")
        dataset_def = create_dataset_def(entities, globals_fields)
        raw_dataset = DictNodeWrapper(input_file.root)
        validate_dataset(raw_dataset, dataset_def)

        dataset_def = create_dataset_index_def(entities)
        dataset = index_tables(raw_dataset, dataset_def)

        output_file = tables.openFile(self.output_path, mode="w")

        globals_table = None
        try:
            if dataset.get('globals', {}).get('periodic') is not None:
                globals_table = dataset.get('globals', {}).get('periodic')
                output_globals = output_file.createGroup("/", "globals",
                                                         "Globals")
                copyTable(input_file.root.globals.periodic, output_file,
                          output_globals)

            entities_tables = dataset['entities']
            output_entities = output_file.createGroup("/", "entities",
                                                      "Entities")
            print " * copying tables"
            for ent_name, entity in entities.iteritems():
                print ent_name, "..."

                # main table

                table = entities_tables[ent_name]

                entity.input_index = table.id2rownum_per_period
                entity.input_rows = table.period_index
                entity.input_table = table.table
                entity.base_period = table.base_period

# this is what should happen
#                entity.indexed_input_table = entities_tables[ent_name]
#                entity.indexed_output_table = entities_tables[ent_name]

                #TODO: copying the table and generally preparing the output
                # file should be a different method than indexing
                print " * copying table..."
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

                output_table = copyTable(table.table,
                                         output_file, output_entities,
                                         entity.fields, stop=stoprow,
                                         show_progress=True)
                entity.output_rows = output_rows
                print "done (%s elapsed)." % time2str(time.time() - start_time)

                print " * building array for first simulated period...",
                start_time = time.time()

                #TODO: this whole process of merging all periods is very
                # opinionated and does not allow individuals to die/disappear
                # before the simulation starts. We couldn't for example,
                # take the output of one of our simulation and
                # re-simulate only some years in the middle, because the dead
                # would be brought back to life. In conclusion, it should be
                # optional.
                entity.array, entity.id_to_rownum = \
                    buildArrayForPeriod(table.table, entity.fields,
                                        entity.input_rows,
                                        entity.input_index, start_period)
                print "done (%s elapsed)." % time2str(time.time() - start_time)
                entity.table = output_table
        except:
            input_file.close()
            output_file.close()
            raise

        return input_file, output_file, globals_table


class Void(DataSource):
    def __init__(self, output_path):
        self.output_path = output_path

    def run(self, globals_fields, entities, start_period):
        output_file = tables.openFile(self.output_path, mode="w")
        output_entities = output_file.createGroup("/", "entities", "Entities")
        for entity in entities.itervalues():
            dtype = np.dtype(entity.fields)
            entity.array = np.empty(0, dtype=dtype)
            entity.id_to_rownum = np.empty(0, dtype=int)
            output_table = output_file.createTable(
                output_entities, entity.name, dtype,
                title="%s table" % entity.name)

            entity.input_table = None
            entity.table = output_table
        return None, output_file, None


def populate_registry(fpath):
    import entities
    import registry
    h5in = tables.openFile(fpath, mode="r")
    for table in h5in.root.entities:
        registry.entity_registry.add(entities.Entity.from_table(table))
    return h5in
