# encoding: utf-8
from __future__ import print_function

import tables

from data import copy_table
from utils import timed

__version__ = "0.4"


def filter_h5(input_path, output_path, condition, copy_globals=True):
    print("filtering for '%s'" % condition)
    input_file = tables.open_file(input_path)
    output_file = tables.open_file(output_path, mode="w")

    # copy globals
    if copy_globals:
        # noinspection PyProtectedMember
        input_file.root.globals._f_copy(output_file.root, recursive=True)

    output_entities = output_file.create_group("/", "entities", "Entities")
    for table in input_file.iterNodes(input_file.root.entities):
        # noinspection PyProtectedMember
        print(table._v_name, "...")
        copy_table(table, output_entities, condition=condition)

    input_file.close()
    output_file.close()


if __name__ == '__main__':
    import sys
    import platform

    print("LIAM HDF5 filter %s using Python %s (%s)\n" %
          (__version__, platform.python_version(), platform.architecture()[0]))

    args = dict(enumerate(sys.argv))
    if len(args) < 4:
        print("""Usage: {} inputpath outputpath condition [copy_globals]
where condition is an expression
      copy_globals is True (default)|False""".format(args[0]))
        sys.exit()

    timed(filter_h5, args[1], args[2], args[3], eval(args.get(4, 'True')))
