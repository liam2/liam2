import tables

from liam2.data import copy_table
from liam2.utils import timed

__version__ = "0.4"


def filter_h5(input_path, output_path, condition, copy_globals=True):
    print(f"filtering for '{condition}'")
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

    py_ver = platform.python_version()
    arch = platform.architecture()[0]
    print(f"LIAM2 HDF5 filter {__version__} using Python {py_ver} ({arch})\n")

    args = dict(enumerate(sys.argv))
    if len(args) < 4:
        print(f"""Usage: {args[0]} inputpath outputpath condition [copy_globals]
where condition is an expression
      copy_globals is True (default)|False""")
        sys.exit()

    timed(filter_h5, args[1], args[2], args[3], eval(args.get(4, 'True')))
