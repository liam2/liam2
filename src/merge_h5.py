from __future__ import print_function

import numpy as np
import tables

from data import merge_arrays, get_fields, index_table_light
from utils import timed, loop_wh_progress, merge_items

__version__ = "0.3"


def get_h5_fields(input_file):
    #noinspection PyProtectedMember
    return dict((table._v_name, get_fields(table))
                for table in input_file.iterNodes(input_file.root.entities))


def merge_h5(input1_path, input2_path, output_path):
    input1_file = tables.open_file(input1_path, mode="r")
    input2_file = tables.open_file(input2_path, mode="r")

    output_file = tables.open_file(output_path, mode="w")

    print("copying globals from", input1_path, end=' ')
    #noinspection PyProtectedMember
    input1_file.root.globals._f_copy(output_file.root, recursive=True)
    print("done.")

    input1_entities = input1_file.root.entities
    input2_entities = input2_file.root.entities

    fields1 = get_h5_fields(input1_file)
    fields2 = get_h5_fields(input2_file)

    ent_names1 = set(fields1.keys())
    ent_names2 = set(fields2.keys())

    output_entities = output_file.create_group("/", "entities", "Entities")
    for ent_name in sorted(ent_names1 | ent_names2):
        print()
        print(ent_name)
        ent_fields1 = fields1.get(ent_name, [])
        ent_fields2 = fields2.get(ent_name, [])
        output_fields = merge_items(ent_fields1, ent_fields2)
        output_table = output_file.create_table(output_entities, ent_name,
                                               np.dtype(output_fields))

        if ent_name in ent_names1:
            table1 = getattr(input1_entities, ent_name)
            print(" * indexing table from %s ..." % input1_path, end=' ')
            input1_rows = index_table_light(table1)
            print("done.")
        else:
            table1 = None
            input1_rows = {}

        if ent_name in ent_names2:
            table2 = getattr(input2_entities, ent_name)
            print(" * indexing table from %s ..." % input2_path, end=' ')
            input2_rows = index_table_light(table2)
            print("done.")
        else:
            table2 = None
            input2_rows = {}

        print(" * merging: ", end=' ')
        input1_periods = input1_rows.keys()
        input2_periods = input2_rows.keys()
        output_periods = sorted(set(input1_periods) | set(input2_periods))

        #noinspection PyUnusedLocal
        def merge_period(period_idx, period):
            if ent_name in ent_names1:
                start, stop = input1_rows.get(period, (0, 0))
                input1_array = table1.read(start, stop)
            else:
                input1_array = None

            if ent_name in ent_names2:
                start, stop = input2_rows.get(period, (0, 0))
                input2_array = table2.read(start, stop)
            else:
                input2_array = None

            if ent_name in ent_names1 and ent_name in ent_names2:
                output_array, _ = merge_arrays(input1_array, input2_array)
            elif ent_name in ent_names1:
                output_array = input1_array
            elif ent_name in ent_names2:
                output_array = input2_array
            else:
                raise Exception("this shouldn't have happened")
            output_table.append(output_array)
            output_table.flush()

        loop_wh_progress(merge_period, output_periods)
        print(" done.")

    input1_file.close()
    input2_file.close()
    output_file.close()


if __name__ == '__main__':
    import sys
    import platform

    print("LIAM HDF5 merge %s using Python %s (%s)\n" % \
          (__version__, platform.python_version(), platform.architecture()[0]))

    args = sys.argv
    if len(args) < 4:
        print("Usage: %s inputpath1 inputpath2 outputpath" % args[0])
        sys.exit()

    timed(merge_h5, args[1], args[2], args[3])
