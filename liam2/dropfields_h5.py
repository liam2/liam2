import tables
import numpy as np

from liam2.data import copy_table, get_fields
from liam2.utils import timed

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
        print(f" * copying table {table._v_name} ({size:.2f} Mb) ...",
              end=' ')
        copy_table(table, output_entities, output_dtype)
        print("done.")

    input_file.close()
    output_file.close()


if __name__ == '__main__':
    import sys
    import platform

    print(f"LIAM2 HDF5 drop fields {__version__} using Python "
          f"{platform.python_version()} ({platform.architecture()[0]})\n")

    args = sys.argv
    if len(args) < 4:
        print(f"Usage: {args[0]} inputpath outputpath field1 [field2 ...]")
        sys.exit()

    timed(dropfields, args[1], args[2], args[3:])
