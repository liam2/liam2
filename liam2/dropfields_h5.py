from __future__ import print_function

import tables
import numpy as np

from data import copy_table, get_fields
from utils import timed

__version__ = "0.1"


def dropfields(input_path, output_path, todrop):
    input_file = tables.open_file(input_path, mode="r")
    input_root = input_file.root

    output_file = tables.open_file(output_path, mode="w")
    output_globals = output_file.create_group("/", "globals", "Globals")

    print(" * copying globals ...", end=' ')
    copy_table(input_root.globals.periodic, output_globals)
    print("done.")

    output_entities = output_file.create_group("/", "entities", "Entities")
    for table in input_file.iterNodes(input_root.entities):
        table_fields = get_fields(table)
        output_dtype = np.dtype([(fname, ftype) for fname, ftype in table_fields
                                 if fname not in todrop])
        size = (len(table) * table.dtype.itemsize) / 1024.0 / 1024.0
        # noinspection PyProtectedMember
        print(" * copying table %s (%.2f Mb) ..." % (table._v_name, size),
              end=' ')
        copy_table(table, output_entities, output_dtype)
        print("done.")

    input_file.close()
    output_file.close()


if __name__ == '__main__':
    import sys
    import platform

    print("LIAM HDF5 drop fields %s using Python %s (%s)\n" % \
          (__version__, platform.python_version(), platform.architecture()[0]))

    args = sys.argv
    if len(args) < 4:
        print("Usage: %s inputpath outputpath field1 [field2 ...]" % args[0])
        sys.exit()

    timed(dropfields, args[1], args[2], args[3:])
