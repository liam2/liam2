import numpy as np
import tables

from liam2.data import index_table_light, get_fields
from liam2.partition import filter_to_indices
from liam2.utils import PrettyTable, merge_items

__version__ = "0.3"
DEFAULT_DIFF_THRESHOLD = 1e-9


def unique_dupes(a):
    is_dupe = np.ones(len(a), dtype=bool)
    unique_indices = np.unique(a, return_index=True)[1]
    is_dupe[unique_indices] = False
    return unique_indices, a[is_dupe]


def diff_array(array1, array2, display_ndiffs=10, raiseondiff=False,
               abs_diff_threshold=1e-9):
    array_max_diff = 0.0
    # use numdiffs=-1 will show all differences
    if len(array1) != len(array2):
        print("length is different: %d vs %d" % (len(array1),
                                                 len(array2)))
        ids1 = array1['id']
        ids2 = array2['id']
        all_ids = np.union1d(ids1, ids2)
        notin1 = np.setdiff1d(ids1, all_ids)
        notin2 = np.setdiff1d(ids2, all_ids)
        if len(notin1):
            print("the following ids are not present in file 1:",
                  notin1)
        elif len(notin2):
            print("the following ids are not present in file 2:",
                  notin2)
        else:
            # some ids must be duplicated
            if len(ids1) > len(all_ids):
                print("file 1 contain duplicate ids:", end=' ')
                uniques, dupes = unique_dupes(ids1)
                print(dupes)
                array1 = array1[uniques]
            if len(ids2) > len(all_ids):
                print("file 2 contain duplicate ids:", end=' ')
                uniques, dupes = unique_dupes(ids2)
                print(dupes)
                array2 = array2[uniques]

    fields1 = get_fields(array1)
    fields2 = get_fields(array2)
    field_names1 = set(array1.dtype.names)
    field_names2 = set(array2.dtype.names)
    # use merge_items instead of field_names1 | field_names2 to preserve
    # ordering
    for field_name, _ in merge_items((fields1, fields2)):
        print("  - %s:" % field_name, end=' ')
        if field_name not in field_names1:
            print("missing in file 1")
            continue
        elif field_name not in field_names2:
            print("missing in file 2")
            continue
        col1, col2 = array1[field_name], array2[field_name]
        if len(col1) != len(col2):
            print("different", end=' ')
            print("(length)")
            continue

        if np.issubdtype(col1.dtype, np.inexact):
            both_nan = np.isnan(col1) & np.isnan(col2)
            strict_eq = both_nan | (col1 == col2)
            # kill all equal values (especially inf and -inf) and nans
            # so that we can compute abs_diff without warning
            col1_forabs = np.where(strict_eq, 0, col1)
            col2_forabs = np.where(strict_eq, 0, col2)
            abs_diff = np.abs(col2_forabs - col1_forabs)
            eq = abs_diff <= abs_diff_threshold
        else:
            strict_eq = col1 == col2
            if col1.dtype.kind == 'b':
                abs_diff = col1 ^ col2
            else:
                abs_diff = np.abs(col2 - col1)
            eq = strict_eq

        field_max_diff = abs_diff.max()
        array_max_diff = max(array_max_diff, field_max_diff)
        if field_max_diff == 0:
            print("ok")
            continue
        elif field_max_diff <= abs_diff_threshold:
            print(f"ok (max diff: {field_max_diff})")
            continue

        print("different", end=' ')
        diff_indices = filter_to_indices(~eq)
        print("(%d differences)" % len(diff_indices))
        ids = array1['id']
        if display_ndiffs:
            filtered_diff_values = abs_diff[diff_indices]
            local_sort_indices = np.argsort(filtered_diff_values,
                                            stable=True)[::-1]
            if display_ndiffs != -1 and len(diff_indices) > display_ndiffs:
                local_sort_indices = local_sort_indices[:display_ndiffs]
            diff_indices = diff_indices[local_sort_indices]
            header_rows = [
                ['id', f'{field_name} (file1)', f'{field_name} (file2)',
                 'abs diff']
            ]
            diff_rows = [
                [ids[idx], str(col1[idx]), str(col2[idx]),
                 str(abs_diff[idx])]
                for idx in diff_indices
            ]
            print(PrettyTable(header_rows + diff_rows))
        if raiseondiff:
            raise Exception('different')
    return array_max_diff


def diff_h5(input1_path, input2_path, numdiffs=10, abs_diff_threshold=1e-9):
    global_max_diff = 0.0
    input1_file = tables.open_file(input1_path, mode="r")
    input2_file = tables.open_file(input2_path, mode="r")

    input1_entities = input1_file.root.entities
    input2_entities = input2_file.root.entities

    # noinspection PyProtectedMember
    ent_names1 = set(table._v_name for table in input1_entities)
    # noinspection PyProtectedMember
    ent_names2 = set(table._v_name for table in input2_entities)
    for ent_name in sorted(ent_names1 | ent_names2):
        print()
        print(ent_name)
        if ent_name not in ent_names1:
            print("missing in file 1")
            continue
        elif ent_name not in ent_names2:
            print("missing in file 2")
            continue

        entity_max_diff = 0.0
        table1 = getattr(input1_entities, ent_name)
        input1_rows = index_table_light(table1)

        table2 = getattr(input2_entities, ent_name)
        input2_rows = index_table_light(table2)

        input1_periods = set(input1_rows.keys())
        input2_periods = set(input2_rows.keys())
        if input1_periods != input2_periods:
            print("periods are different in both files for '%s'" % ent_name)
            print("periods only in file1:",
                  sorted(input1_periods - input2_periods))
            print("periods only in file2:",
                  sorted(input2_periods - input1_periods))

        for period in sorted(input1_periods & input2_periods):
            print("* period:", period)
            start, stop = input1_rows.get(period, (0, 0))
            array1 = table1.read(start, stop)

            start, stop = input2_rows.get(period, (0, 0))
            array2 = table2.read(start, stop)

            max_diff = diff_array(array1, array2, numdiffs,
                                  abs_diff_threshold=abs_diff_threshold)
            print(f"max absolute difference for '{ent_name}' in {period}:", max_diff)
            entity_max_diff = max(entity_max_diff, max_diff)
        print(f"max absolute difference for '{ent_name}':", entity_max_diff)
        global_max_diff = max(global_max_diff, entity_max_diff)
    input1_file.close()
    input2_file.close()
    print(f"max absolute difference overall:", global_max_diff)
    return global_max_diff


if __name__ == '__main__':
    import sys
    import platform

    print("LIAM HDF5 diff %s using Python %s (%s)\n" % \
          (__version__, platform.python_version(), platform.architecture()[0]))

    args = sys.argv
    if len(args) < 3:
        print(f"""\
Usage: {args[0]} inputpath1 inputpath2 [numdiffs] [abs_diff_threshold]
  where numdiffs defaults to 10 (use -1 to show all differences)
    and abs_diff_threshold defaults to {DEFAULT_DIFF_THRESHOLD}""")
        sys.exit()

    if len(args) > 3:
        numdiffs = int(args[3])
    else:
        numdiffs = 10
    if len(args) > 4:
        abs_diff_threshold = float(args[4])
    else:
        abs_diff_threshold = DEFAULT_DIFF_THRESHOLD
    diff_h5(args[1], args[2], numdiffs, abs_diff_threshold=abs_diff_threshold)
