import numpy as np
import tables

from data import copyTable, appendTable, get_fields, table_size
from utils import timed

__version__ = "0.2"

def get_fields(input_file):
    input_entities = input_file.root.entities
    fields = {}
    for table in input_file.iterNodes(input_entities):
        table_fields = get_fields(table)
        table_name = table._v_name 
        if table_name.endswith("_per_period"):
            ent_name = table_name[:-11] 
            fields.setdefault(ent_name, {})['period'] = table_fields
        else:
            fields.setdefault(table_name, {})['main'] = table_fields
    return fields

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
    
    ent_names1 = set([table._v_name
                      for table in input1_file.iterNodes(input1_entities)
                      if not table._v_name.endswith("_per_period")])
    ent_names2 = set([table._v_name
                      for table in input2_file.iterNodes(input2_entities)
                      if not table._v_name.endswith("_per_period")])
    ent_names = sorted(ent_names1 | ent_names2)
    
    fields1 = get_fields(input1_file)
    fields2 = get_fields(input2_file)
    
    output_entities = output_file.createGroup("/", "entities", "Entities")
    missing_entity_fields = {'main': [], 'period': []}
    filters = tables.Filters(complevel=5, complib='zlib',
                             fletcher32=True)
    for ent_name in ent_names:
        print ent_name
        ent_fields1 = fields1.get(ent_name, missing_entity_fields)
        ent_fields2 = fields2.get(ent_name, missing_entity_fields)
        main_fields = merge_fields(ent_fields1['main'], ent_fields2['main'])
        period_fields = merge_fields(ent_fields1['period'], 
                                     ent_fields2['period'])
        
        if ent_name in ent_names1:
            table = getattr(input1_entities, ent_name)
            print " * copying main table from", input1_path, \
                  "(%.2f Mb)" % table_size(table)
            output_table = copyTable(table, output_file, output_entities, 
                                     main_fields, filters=filters)
            print " done.\n"

            print " * copying period table from", input1_path
            per_period_table = getattr(input1_entities, ent_name + "_per_period")
            output_per_period_table = copyTable(per_period_table, output_file, 
                                                output_entities, period_fields,
                                                filters=filters)
            print " done.\n"
        else:
            output_table = output_file.createTable(output_entities, ent_name, 
                                                   np.dtype(main_fields),
                                                   filters=filters)
            output_per_period_table = \
                output_file.createTable(output_entities,
                                        ent_name + "_per_period", 
                                        np.dtype(period_fields),
                                        filters=filters)
        
        if ent_name in ent_names2:
            table = getattr(input2_entities, ent_name)
            print " * copying main table from", input2_path, \
                  "(%.2f Mb)" % table_size(table)
            appendTable(table, output_table)
            print " done.\n"
            
            print " * copying period table from", input2_path
            per_period_table = getattr(input2_entities, ent_name + "_per_period")
            appendTable(per_period_table, output_per_period_table)
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
