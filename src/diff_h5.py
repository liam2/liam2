import numpy as np
import tables

from data import get_fields, index_table_light
from utils import PrettyTable

__version__ = "0.2"


def get_h5_fields(input_file):
    return dict((table._v_name, get_fields(table))
                for table in input_file.iterNodes(input_file.root.entities))


def unique_dupes(a):
    is_dupe = np.ones(len(a), dtype=bool)
    unique_indices = np.unique(a, return_index=True)[1]
    is_dupe[unique_indices] = False
    return unique_indices, a[is_dupe]


def diff_h5(input1_path, input2_path, numdiff=10):
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
    for ent_name in sorted(ent_names1 | ent_names2):
        print
        print ent_name
        if ent_name not in ent_names1:
            print "missing in file 1"
            continue
        elif ent_name not in ent_names2:
            print "missing in file 2"
            continue

        ent_fields1 = fields1.get(ent_name, [])
        ent_fields2 = fields2.get(ent_name, [])
        fnames1 = set(fname for fname, _ in ent_fields1)
        fnames2 = set(fname for fname, _ in ent_fields2)

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
            array1 = table1.read(start, stop)

            start, stop = input2_rows.get(period, (0, 0))
            array2 = table2.read(start, stop)

            if len(array1) != len(array2):
                print "length is different: %d vs %d" % (len(array1),
                                                         len(array2))
                ids1 = array1['id']
                ids2 = array2['id']
                all_ids = np.union1d(ids1, ids2)
                notin1 = np.setdiff1d(ids1, all_ids)
                notin2 = np.setdiff1d(ids2, all_ids)
                if notin1:
                    print "the following ids are not present in file 1:", \
                          notin1
                elif notin2:
                    print "the following ids are not present in file 2:", \
                          notin2
                else:
                    # some ids must be duplicated
                    if len(ids1) > len(all_ids):
                        print "file 1 contain duplicate ids:",
                        uniques, dupes = unique_dupes(ids1)
                        print dupes
                        array1 = array1[uniques]
                    if len(ids2) > len(all_ids):
                        print "file 2 contain duplicate ids:",
                        uniques, dupes = unique_dupes(ids2)
                        print dupes
                        array2 = array2[uniques]

            for fname in sorted(fnames1 | fnames2):
                print "  - %s:" % fname,
                if fname not in fnames1:
                    print "missing in file 1"
                    continue
                elif fname not in fnames2:
                    print "missing in file 2"
                    continue
                col1, col2 = array1[fname], array2[fname]
                if np.array_equal(col1, col2):
                    print "ok"
                else:
                    print "different",
                    if len(col1) != len(col2):
                        print "(length)"
                    else:
                        diff = (col1 != col2).nonzero()[0]
                        print "(%d differences)" % len(diff)
                        ids = array1['id']
                        if len(diff) > numdiff:
                            diff = diff[:numdiff]
                        print PrettyTable([['id',
                                            fname + ' (file1)',
                                            fname + ' (file2)']] +
                                          [[ids[idx], col1[idx], col2[idx]]
                                           for idx in diff])

    input1_file.close()
    input2_file.close()


if __name__ == '__main__':
    import sys
    import platform

    print "LIAM HDF5 diff %s using Python %s (%s)\n" % \
          (__version__, platform.python_version(), platform.architecture()[0])

    args = sys.argv
    if len(args) < 3:
        print "Usage: %s inputpath1 inputpath2 [numdiff]" % args[0]
        sys.exit()

    if len(args) > 3:
        numdiff = int(args[3])
    else:
        numdiff = 10
    diff_h5(args[1], args[2], numdiff)
