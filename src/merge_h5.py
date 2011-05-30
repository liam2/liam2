import numpy as np
import tables

from data import copyTable, appendTable, get_fields, table_size
from utils import timed

__version__ = "0.2"

def get_h5_fields(input_file):
    return dict((table._v_name, get_fields(table)) 
                for table in input_file.iterNodes(input_file.root.entities))

def merge_fields(fields1, fields2):
    names1 = set(name for name, _ in fields1)
    names2 = set(name for name, _ in fields2)
    names_notin1 = names2 - names1
    fields_notin1 = [(name, type_) for name, type_ in fields2 
                     if name in names_notin1]
    return fields1 + fields_notin1

def merge_h5(input1_path, input2_path, output_path):        
    input1_file = tables.openFile(input1_path, mode="r")
    input2_file = tables.openFile(input2_path, mode="r")
    input1_root = input1_file.root
    input2_root = input2_file.root
    
    output_file = tables.openFile(output_path, mode="w")
    output_globals = output_file.createGroup("/", "globals", "Globals")

    print
    print "copying globals from", input1_path
    copyTable(input1_root.globals.periodic, output_file, output_globals)
    print " done.\n"
    
    input1_entities = input1_root.entities
    input2_entities = input2_root.entities
    
    fields1 = get_h5_fields(input1_file)
    fields2 = get_h5_fields(input2_file)

    ent_names1 = set(fields1.keys())
    ent_names2 = set(fields2.keys())
    
    output_entities = output_file.createGroup("/", "entities", "Entities")
    for ent_name in sorted(ent_names1 | ent_names2):
        print ent_name
        ent_fields1 = fields1.get(ent_name, [])
        ent_fields2 = fields2.get(ent_name, [])
        output_fields = merge_fields(ent_fields1, ent_fields2)
        
        if ent_name in ent_names1:
            table = getattr(input1_entities, ent_name)
            print " * copying table from %s (%.2f Mb)" % (input1_path, 
                                                          table_size(table))
            output_table = copyTable(table, output_file, output_entities, 
                                     output_fields)
            print " done.\n"
        else:
            output_table = output_file.createTable(output_entities, ent_name, 
                                                   np.dtype(output_fields))

        if ent_name in ent_names2:
            table = getattr(input2_entities, ent_name)
            print " * copying table from %s (%.2f Mb)" % (input2_path,
                                                          table_size(table))
            appendTable(table, output_table)
            print " done.\n"

    input1_file.close()
    input2_file.close()
    output_file.close()


if __name__ == '__main__':
    import sys, platform

    print "LIAM HDF5 merge %s using Python %s (%s)\n" % \
          (__version__, platform.python_version(), platform.architecture()[0])

    args = sys.argv
    if len(args) < 4:
        print "Usage: %s inputpath1 inputpath2 outputpath" % args[0]
        sys.exit()
    
    timed(merge_h5, args[1], args[2], args[3])
