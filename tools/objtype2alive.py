import sys

from data_main import load_objtype, export_csv_3col
import numpy as np

if __name__ == '__main__':
    print 'Using Python %s' % sys.version
    
    args = sys.argv
    if len(args) < 2:
        print "Usage: %s [entity_name]" % args[0]
        sys.exit()
    else:
        ent_name = args[1]
    
    
    ids = load_objtype("objtype_%s.txt" % ent_name)
    dtype = np.dtype([('period', int), ('id', int), ('alive', int)])
    data = np.empty(len(ids), dtype=dtype)
    data['id'] = ids
    data['period'] = 2002
    data['alive'] = 1 
    
    export_csv_3col("%s_co_alive.txt" % ent_name, data, 'alive', "\t")
    print "done"