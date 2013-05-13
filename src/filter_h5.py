from __future__ import print_function

import tables

from data import copyTable
from utils import timed

__version__ = "0.3"


def filter_h5(input_path, output_path, condition):
    print("filtering for '%s'" % condition)
    input_file = tables.openFile(input_path, mode="r")
    output_file = tables.openFile(output_path, mode="w")

    # copy globals
    input_file.root.globals._f_copy(output_file.root, recursive=True)

    output_entities = output_file.createGroup("/", "entities", "Entities")
    for table in input_file.iterNodes(input_file.root.entities):
        print(table._v_name, "...")
        copyTable(table, output_entities, condition=condition)

    input_file.close()
    output_file.close()


if __name__ == '__main__':
    import sys
    import platform

    print("LIAM HDF5 filter %s using Python %s (%s)\n" % \
          (__version__, platform.python_version(), platform.architecture()[0]))

    args = sys.argv
    if len(args) < 4:
        print("Usage: %s inputpath outputpath condition" % args[0])
        sys.exit()

    timed(filter_h5, args[1], args[2], args[3])
