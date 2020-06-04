# encoding: utf-8
from __future__ import absolute_import, division, print_function

from os.path import splitext, basename
import time

import tables
import numpy as np

from liam2.simulation import Simulation
from liam2.utils import timed, time2str, multi_get, MB

__version__ = "0.3"


def get_shrink_dict(values, shuffle=False):
    """

    Parameters
    ----------
    values : iterable of values
        each value can be anything

    Returns
    -------
    dict
        {value: sorted_unique_value_num}

    Examples
    --------
    >>> m = get_shrink_dict(['a', 'c', 'b', 'a'])
    >>> sorted(m.items())
    [('a', 0), ('b', 1), ('c', 2)]
    """
    unique_values = np.unique(values)
    if shuffle:
        np.random.shuffle(unique_values)
    return {value: i for i, value in enumerate(unique_values)}


def table_empty_like(input_table, new_parent=None, new_name=None):
    if new_parent is None:
        new_parent = input_table._v_parent

    if new_name is None:
        new_name = input_table.name
    assert isinstance(new_parent, tables.Group)
    output_file = new_parent._v_file
    # TODO: copy other table attributes
    return output_file.create_table(new_parent, new_name,
                                    input_table.dtype,
                                    title=input_table._v_title)


def table_apply_map(input_table, new_parent, fields_maps):
    """Copy the contents of a table to another table passing some fields
    through a mapping.

    Parameters
    ----------
    input_table : tables.Table
    new_parent : tables.Group
    fields_maps : {fname: {value: new_value}}
        fields for which there is no map are copied unmodified, but for
        fields with a map, all possible values must be present in the map.
    """
    assert isinstance(input_table, tables.Table)
    assert isinstance(new_parent, tables.Group)
    output_table = table_empty_like(input_table, new_parent)
    # TODO: it would probably be faster to use a buffer (load several rows at
    #       once) and apply map using np.vectorize (or use a pd.Index)
    for i, row in enumerate(input_table):
        newrow = output_table.row
        rec = row.fetch_all_fields()
        for fname in rec.dtype.fields:
            value = rec[fname]
            if fname in fields_maps:
                value_map = fields_maps[fname]
                try:
                    value = value_map[value]
                except KeyError:
                    print("WARNING: row {row} has {fname} == {value} and this does not correspond to an existing id "
                          "in the '{table}' table, this value has not been modified !"
                          .format(row=i, fname=fname, value=value, table=input_table.name))
            newrow[fname] = value
        newrow.append()
        if i % 1000 == 0:
            output_table.flush()
    output_table.flush()


import csv
import tempfile
import os

def csv_rows(filename):
    with open(filename) as fp:
        yield from csv.reader(fp)


def mergesort(sorted_filenames, columns, nway=2):
    """Merge sorted csv files into a single output file"""

    keyfunc = operator.itemgetter(*columns)

    while len(sorted_filenames) > 1:
        merge_filenames, sorted_filenames = sorted_filenames[:nway], sorted_filenames[nway:]
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as output_fp:
            writer = csv.writer(output_fp)
            for row in heapq.merge(*[csv_rows(filename) for filename in merge_filenames],
                                   key=keyfunc):
                writer.writerow(row)
            sorted_filenames.append(output_fp.name)
        for filename in merge_filenames:
            os.remove(filename)
    return sorted_filenames[0]


MB = 2 ** 20

import operator
import heapq
import numpy as np
import math


def sort_table(input_table, col_names, new_parent=None, new_name=None, nway=2, buffersize=10 * MB):
    """Sort a potentially huge table using col_names"""

    if new_parent is None:
        new_parent = input_table._v_parent

    output_file = new_parent._v_file
    dtype = input_table.dtype
    max_buffer_rows = buffersize // dtype.itemsize
    table_name = input_table.name
    if new_name is None:
        new_name = table_name

    # read chunks, sort them and store them back in temporary tables
    remaining_rows = len(input_table)
    ndigits = math.ceil(math.log10(remaining_rows + 1))
    sorted_tables = []
    buffer_rows = max_buffer_rows
    buffer = np.empty(buffer_rows, dtype=dtype)
    start = 0
    while remaining_rows > 0:
        buffer_rows = min(remaining_rows, max_buffer_rows)
        if buffer_rows < len(buffer):
            # last chunk is smaller
            buffer = np.empty(buffer_rows, dtype=dtype)
        stop = start + buffer_rows
        input_table.read(start, stop, out=buffer)
        buffer.sort(order=col_names)
        chunk_name = f"_{table_name}_sorted_{start:0{ndigits}}_to_{stop - 1:0{ndigits}}"
        sorted_chunk_table = output_file.create_table(new_parent, chunk_name, dtype)
        sorted_chunk_table.append(buffer)
        sorted_chunk_table.flush()
        sorted_tables.append(sorted_chunk_table)
        start += buffer_rows
        remaining_rows -= buffer_rows

    all_col_names = dtype.names
    col_indices = [all_col_names.index(name) for name in col_names]
    keyfunc = operator.itemgetter(*col_indices)

    if nway is None:
        nway = len(sorted_tables)

    iter_num = 0
    remaining_tables = sorted_tables
    while len(remaining_tables) > 1:
        current_tables, remaining_tables = remaining_tables[:nway], remaining_tables[nway:]

        # TODO: copy other table attributes
        table = output_file.create_table(new_parent, f'_{table_name}_merged_{iter_num}', dtype,
                                         title=input_table._v_title)
        chunk_lengths = [len(table) for table in current_tables]
        remaining_rows = sum(chunk_lengths)

        def yield_row_tuples(table):
            for row in table.iterrows():
                yield row.fetch_all_fields()

        merged_rows_iter = heapq.merge(*[yield_row_tuples(table) for table in current_tables],
                                       key=keyfunc)
        while remaining_rows > 0:
            buffer_rows = min(remaining_rows, max_buffer_rows)
            array = np.fromiter(merged_rows_iter, dtype=dtype, count=buffer_rows)
            remaining_rows -= buffer_rows
            assert len(array) > 0
            table.append(array)
            table.flush()

        remaining_tables.append(table)

        for table in current_tables:
            output_file.remove_node(table)

        iter_num += 1

    merged_table = remaining_tables[0]
    output_file.rename_node(merged_table, new_name, overwrite=True)
    return merged_table


def table_sort(input_table, fnames, new_parent=None, new_name=None, buffersize=10 * MB):
    """Copy the contents of a table to another table sorting rows along
    several columns.

    Parameters
    ----------
    input_table : tables.Table
    new_parent : tables.Group
    fnames : iterable
        field names. Sort by first field then second.
        eg ['period', 'id']
    """
    assert isinstance(input_table, tables.Table)

    output_table = table_empty_like(input_table, new_parent, new_name)
    sort_columns = [input_table.col(fname) for fname in fnames]
    dtype = input_table.dtype
    # TODO: try with setting a pytables index and using read_sorted (cannot work!)
    indices = np.lexsort(sort_columns[::-1])
    max_buffer_rows = buffersize // dtype.itemsize
    lines_left = len(indices)
    buffer_rows = min(lines_left, max_buffer_rows)
    start, stop = 0, buffer_rows
    while lines_left > 0:
        buffer_rows = min(lines_left, max_buffer_rows)
        chunk_indices = indices[start:stop]
        # out is not supported in read_coordinates so far
        chunk = input_table.read_coordinates(chunk_indices)
        output_table.append(chunk)
        # TODO: try flushing after each chunk, this should reduce memory
        # use on large models, and (hopefully) should not be much slower
        # given our chunks are rather large
        # >>> on our 300k sample, it does not seem to make any difference
        #     either way. I'd like to test this on the 2000k sample, but
        #     that will have to wait for 0.8
        lines_left -= buffer_rows
        start += buffer_rows
        stop += buffer_rows
    output_table.flush()


def h5_apply_func(input_path, output_path, node_func):
    """
    Apply node_func to all nodes of input_path and store the result in
    output_path

    Parameters
    ----------
    input_path : str
        path to .h5 input file
    output_path : str
        path to .h5 output file
    node_func : function
        function that will be applied to all nodes
        func(node, new_parent) -> new_node
        new_node must be node if node must be copied
                         None if node must not be copied
                         another Node if node must not be copied (was already
                                      handled/copied/modified by func)
    """
    with tables.open_file(input_path) as input_file, \
            tables.open_file(output_path, mode="w") as output_file:
        for node in input_file.walk_nodes(classname='Leaf'):
            if node is not input_file.root:
                print(node._v_pathname, "...", end=' ')
                parent_path = node._v_parent._v_pathname
                if parent_path in output_file:
                    new_parent = output_file.get_node(parent_path)
                else:
                    new_parent = output_file._create_path(parent_path)
                new_node = node_func(node, new_parent)
                if new_node is node:
                    print("copying (without modifications) ...", end=' ')
                    node._f_copy(new_parent)
                print("done.")


# def h5_copy(input_path, output_path):
#     def copy_node(node, new_parent):
#         return node
#     return h5_apply_func(input_path, output_path, copy_node)
#
#
# def h5_apply_flat_map(input_path, output_path, changes):
#     def map_node(node, new_parent):
#         node_changes = changes.get(node._v_pathname)
#         if node_changes is not None:
#             return map_table(node, new_parent, node_changes)
#         else:
#             return node
#     return h5_apply_func(input_path, output_path, map_node)


def h5_apply_rec_func(input_path, output_path, change_funcs):
    def handle_node(node, new_parent):
        func = multi_get(change_funcs, node._v_pathname.lstrip('/'))
        if func is not None:
            return func(node, new_parent)
        else:
            return node
    return h5_apply_func(input_path, output_path, handle_node)


def h5_apply_rec_map(input_path, output_path, changes):
    # we could also
    # transform {l1: {l2: ... {lN: value_map}}}
    #        to {l1: {l2: ... {lN: func(n, p): map_table(n, p, value_map)}}}
    # then use h5_apply_rec_func
    def handle_node(node, new_parent):
        node_changes = multi_get(changes, node._v_pathname.lstrip('/'))
        if node_changes is not None:
            print("applying changes ...", end=' ')
            return table_apply_map(node, new_parent, node_changes)
        else:
            return node
    return h5_apply_func(input_path, output_path, handle_node)


def change_ids(input_path, output_path, changes, shuffle=False):
    with tables.open_file(input_path) as input_file:
        input_entities = input_file.root.entities
        print(" * indexing entities tables")
        idmaps = {}
        for ent_name in changes.keys():
            print("    -", ent_name, "...", end=' ')
            start_time = time.time()
            table = getattr(input_entities, ent_name)

            new_ids = get_shrink_dict(table.col('id'), shuffle=shuffle)
            if -1 in new_ids:
                raise Exception('found id == -1 in %s which is invalid (only link '
                                'columns can be -1)' % ent_name)
            # -1 links should stay -1
            new_ids[-1] = -1
            idmaps[ent_name] = new_ids
            print("done (%s elapsed)." % time2str(time.time() - start_time))

    print(" * modifying ids")
    fields_maps = {ent_name: {'id': idmaps[ent_name]}
                   for ent_name in changes}
    for ent_name, fields in changes.items():
        for target_ent, fname in fields:
            fields_maps[ent_name][fname] = idmaps[target_ent]
    h5_apply_rec_map(input_path, output_path, {'entities': fields_maps})


def h5_sort(input_path, output_path, entities=None):
    """
    Sort the tables of a list of entities by period, then id

    Parameters
    ----------
    input_path : str
    output_path : str
    entities : list of str
        names of entities to sort
    """

    def sort_entity(table, new_parent):
        print("sorting ...", end=' ')
        return table_sort(table, ('period', 'id'), new_parent)

    if entities is None:
        with tables.open_file(input_path) as f:
            entities = list(f.root.entities._v_children.keys())
    print(" * sorting entities tables")
    to_sort = {'entities': {ent_name: sort_entity for ent_name in entities}}
    h5_apply_rec_func(input_path, output_path, to_sort)


def fields_from_entity(entity):
    """
    Parameters
    ----------
    entity : liam2.entities.Entity

    Returns
    -------
    list
        [(target_ent1, fname1), (target_ent2, fname2)]
    """
    return [(link._target_entity_name, link._link_field)
            for link in entity.links.values()]


if __name__ == '__main__':
    with tables.open_file('c:/tmp/is/flanders.h5', 'a') as t:
        sort_table(t.root.entities.person, ['period', 'id'], buffersize=10000)

    import sys
    import platform

    print("LIAM2 HDF5 idchanger %s using Python %s (%s)\n" %
          (__version__, platform.python_version(), platform.architecture()[0]))

    args = sys.argv
    if len(args) < 4:
        print("""Usage: {} action inputpath outputpath [link_fields|entities]
    where:
      * action must be either 'shuffle', 'shrink' or 'sort'.
        'shuffle' will randomize ids (but keep links consistent)
        'shrink' will make ids as small as possible (e.g. if they range from
            1000 to 1999, they will be changed to 0 to 999.
        'sort' will sort rows by 'period' then 'id'.
      * inputpath can point to either a .yml simulation file or a .h5 file.
        If a .h5 file is supplied, link_fields must be supplied for shuffle or
        shrink, otherwise link_fields or entities are taken from the
        simulation file.
      * outputpath is the path to the .h5 output file
      * link_fields, if given should have the following format. [] denote
        optional parts:
        entityname1:[target_entity.]linkfield1,linkfield2;entityname2:...
      * entities, if given should have the following format:
        entityname1,entityname2,...
        if not given, all entities present in the file will be sorted.
""".format(basename(args[0])))
        sys.exit()

    action = args[1]
    inputpath = args[2]
    outputpath = args[3]

    _, ext = splitext(inputpath)
    if ext in ('.h5', '.hdf5'):
        if action != 'sort':
            if len(args) < 5 and action != 'sort':
                print("link_fields argument must be provided if using an .h5 "
                      "input file")

            entities = [entity.split(':') for entity in args[4].split(';')]
            to_change = {ent_name: fields.split(',')
                         for ent_name, fields in entities}
            # convert {ent_name: [target_ent1.fname1, target_ent2.fname2]}
            #      to {ent_name: [(target_ent1, fname1), (target_ent2, fname2)]}
            for ent_name, fields in to_change.items():
                for i, fname in enumerate(fields):
                    fields[i] = \
                        fname.split('.') if '.' in fname else (ent_name, fname)
    else:
        simulation = Simulation.from_yaml(inputpath)
        inputpath = simulation.data_source.input_path
        to_change = {entity.name: fields_from_entity(entity)
                     for entity in simulation.entities}

    assert action in {'shrink', 'shuffle', 'sort'}
    if action == 'shrink':
        timed(change_ids, inputpath, outputpath, to_change)
    elif action == 'shuffle':
        timed(change_ids, inputpath, '_shuffled_temp.h5', to_change,
              shuffle=True)
        timed(h5_sort, '_shuffled_temp.h5', outputpath, list(to_change.keys()))
    else:
        ent_names = args[4].split(',') if len(args) >= 5 else None
        timed(h5_sort, inputpath, outputpath, ent_names)
