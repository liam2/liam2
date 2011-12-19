import tables

from data import copyTable, get_fields, table_size
from utils import timed

__version__ = "0.1"


def dropfields(input_path, output_path, todrop):
    input_file = tables.openFile(input_path, mode="r")
    input_root = input_file.root

    output_file = tables.openFile(output_path, mode="w")
    output_globals = output_file.createGroup("/", "globals", "Globals")

    print " * copying globals ...",
    copyTable(input_root.globals.periodic, output_file, output_globals)
    print "done."

    output_entities = output_file.createGroup("/", "entities", "Entities")
    for table in input_file.iterNodes(input_root.entities):
        table_fields = get_fields(table)
        table_fields = [(fname, ftype) for fname, ftype in table_fields
                        if fname not in todrop]
        print " * copying table %s (%.2f Mb) ..." % (table._v_name,
                                                     table_size(table)),
        copyTable(table, output_file, output_entities,
                  table_fields)
        print "done."

    input_file.close()
    output_file.close()


if __name__ == '__main__':
    import sys
    import platform

    print "LIAM HDF5 drop fields %s using Python %s (%s)\n" % \
          (__version__, platform.python_version(), platform.architecture()[0])

    args = sys.argv
    if len(args) < 4:
        print "Usage: %s inputpath outputpath field1 [field2 ...]" % args[0]
        sys.exit()

    timed(dropfields, args[1], args[2], args[3:])
