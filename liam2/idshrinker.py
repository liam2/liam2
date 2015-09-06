# encoding: utf-8
from __future__ import print_function

import tables
import numpy as np
import time

from data import copy_table
from utils import timed, time2str

__version__ = "0.1"


def index_table(table):
    """
    table is an iterable of rows, each row is a mapping (name -> value).
    Rows must contain at least an 'id' column.
    """
    unique_ids = np.unique(table.col('id'))
    return {identifier: i for i, identifier in enumerate(unique_ids)}


def map_rows(input_table, output_table, mappings):
    newrow = output_table.row
    for i, row in enumerate(input_table):
        rec = row.fetch_all_fields()
        for fname in rec.dtype.fields:
            value = rec[fname]
            if fname in mappings:
                value = mappings[fname].get(value, value)
            newrow[fname] = value
        newrow.append()
        if i % 1000 == 0:
            output_table.flush()
    output_table.flush()


def map_file(input_file, output_file, entities_map):
    # copy globals
    if hasattr(input_file.root, 'globals'):
        # noinspection PyProtectedMember
        input_file.root.globals._f_copy(output_file.root, recursive=True)

    print(" * copying tables")
    output_entities = output_file.create_group("/", "entities", "Entities")
    for table in input_file.iterNodes(input_file.root.entities):
        # noinspection PyProtectedMember
        ent_name = table._v_name
        print(ent_name, "...")
        if ent_name in entities_map:
            # noinspection PyProtectedMember
            output_table = output_file.create_table(output_entities, table.name,
                                                    table.dtype,
                                                    title=table._v_title)
            map_rows(table, output_table, entities_map[ent_name])
        else:
            copy_table(table, output_entities)


def shrinkids(input_path, output_path, toshrink):
    input_file = tables.open_file(input_path)
    output_file = tables.open_file(output_path, mode="w")
    input_entities = input_file.root.entities
    print(" * indexing tables")
    idmaps = {}
    for ent_name, fields in toshrink.iteritems():
        print("    -", ent_name, "...", end=' ')
        start_time = time.time()
        idmaps[ent_name] = index_table(getattr(input_entities, ent_name))
        print("done (%s elapsed)." % time2str(time.time() - start_time))

    fields_to_change = {ent_name: {'id': idmaps[ent_name]}
                        for ent_name in toshrink}
    for ent_name, fields in toshrink.iteritems():
        # fields_to_change[ent_name] = d = []
        for fname in fields:
            if '.' in fname:
                source_ent, fname = fname.split('.')
            else:
                source_ent = ent_name
            fields_to_change[source_ent][fname] = idmaps[ent_name]
    print(" * shrinking ids")
    map_file(input_file, output_file, fields_to_change)
    input_file.close()
    output_file.close()


def fixlinks(input_path, output_path, tofix):
    input_file = tables.open_file(input_path)
    output_file = tables.open_file(output_path, mode="w")
    tochange = {ent_name: {fname: {0: -1} for fname in fnames}
                for ent_name, fnames in tofix.iteritems()}
    print(" * fixing links")
    map_file(input_file, output_file, tochange)
    input_file.close()
    output_file.close()


if __name__ == '__main__':
    import sys
    import platform

    print("LIAM2 HDF5 idshrinker %s using Python %s (%s)\n" %
          (__version__, platform.python_version(), platform.architecture()[0]))

    args = sys.argv
    if len(args) < 4:
        print("Usage: %s inputpath outputpath toshrink" % args[0])
        print("    where toshrink is: entityname1:[entityname2.]linkfield1,"
              "linkfield2;entityname2:...")
        sys.exit()

    entities = [entity.split(':') for entity in args[3].split(';')]
    toshrink = {ent_name: fields.split(',') for ent_name, fields in entities}
    timed(shrinkids, args[1], 'shrinkedids.h5', toshrink)
    timed(fixlinks, 'shrinkedids.h5', args[2], toshrink)