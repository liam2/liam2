import numpy as np
import tables

from data import get_fields, index_table_light

__version__ = "0.1"


def get_h5_fields(input_file):
    return dict((table._v_name, get_fields(table))
                for table in input_file.iterNodes(input_file.root.entities))


def diff_h5(input1_path, input2_path):
    input1_file = tables.openFile(input1_path, mode="r")
    input2_file = tables.openFile(input2_path, mode="r")

#    print "copying globals from", input1_path,
#    input1_file.root.globals._f_copy(output_file.root, recursive=True)
#    print "done."

    input1_entities = input1_file.root.entities
    input2_entities = input2_file.root.entities

    fields1 = get_h5_fields(input1_file)
    fields2 = get_h5_fields(input2_file)

    ent_names1 = set(fields1.keys())
    ent_names2 = set(fields2.keys())
    if ent_names1 != ent_names2:
        raise Exception("entities are different in both files")

    for ent_name in sorted(ent_names1 | ent_names2):
        print
        print ent_name
        ent_fields1 = fields1.get(ent_name, [])
        ent_fields2 = fields2.get(ent_name, [])
        fnames1 = set(fname for fname, ftype in ent_fields1)
        fnames2 = set(fname for fname, ftype in ent_fields2)
        
        table1 = getattr(input1_entities, ent_name)
        input1_rows = index_table_light(table1)

        table2 = getattr(input2_entities, ent_name)
        input2_rows = index_table_light(table2)

        input1_periods = input1_rows.keys()
        input2_periods = input2_rows.keys()
        if input1_periods != input2_periods:
            print "periods are different in both files for '%s'" % ent_name
            
        for period in sorted(set(input1_periods) & set(input2_periods)):
            print "* period:", period
            start, stop = input1_rows.get(period, (0, 0))
            input1_array = table1.read(start, stop)

            start, stop = input2_rows.get(period, (0, 0))
            input2_array = table2.read(start, stop)
            
            for fname in sorted(fnames1 | fnames2):
                print "  - %s:" % fname,
                if fname not in fnames1:
                    print "missing in file 1"
                    continue
                elif fname not in fnames2:
                    print "missing in file 2"
                    continue
                data1, data2 = input1_array[fname], input2_array[fname]
                if np.array_equal(data1, data2):
                    print "ok"
                else:    
                    print "different"

    input1_file.close()
    input2_file.close()


if __name__ == '__main__':
    import sys
    import platform

    print "LIAM HDF5 diff %s using Python %s (%s)\n" % \
          (__version__, platform.python_version(), platform.architecture()[0])

    args = sys.argv
    if len(args) < 3:
        print "Usage: %s inputpath1 inputpath2" % args[0]
        sys.exit()

    diff_h5(args[1], args[2])
