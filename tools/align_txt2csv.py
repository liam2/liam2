import sys
import csv
from itertools import izip
from os import path 

def rename_var(name):
    if name.startswith('p_'):
        name = name[2:]

#    if name.startswith('co') and not name.startswith('coef'):
#        name = name[2:]
#        if name[0] == '_':
#            name = name[1:]
#    if name == 'year':
    if name == 'coyear':
        return 'period'
    return name

def load_txt_align(input_path, inverted=False):
    start_col = 2 + int(not inverted)
             
    with open(input_path, "rb") as f:
        lines = list(csv.reader(f, delimiter='\t'))

    assert lines[15][0] == 'name1'
    var1 = lines[15][1]
    assert lines[16][0] == 'name2'
    var2 = lines[16][1]
    
    assert lines[17][0] == 'nCategory1'
    ncateg1 = int(lines[17][1])
    assert lines[18][0] == 'nCategory2'
    ncateg2 = int(lines[18][1])
    
    assert lines[21][0] == 'predict'
    # skip the 21 first lines
    lines = lines[22:]
    # colval        2003    2003    2004    2004    2005    2005    ...
    colvals = lines.pop(0)
    assert colvals.pop(0) == 'colval'
    assert colvals.pop(0) == ''
    
    # discard any extra data at the end of the line
    colvals = [eval(val) for val in colvals[:ncateg2*2:2] if val]
    assert len(colvals) == ncateg2
    
    # skip next two lines
    assert lines.pop(0)[0] == 'colvars'
    assert lines.pop(0)[:2] == ['rowval', 'rowvars']
    
    rowvals = [eval(line[1]) for line in lines[:ncateg1]]
    assert len(rowvals) == ncateg1
    
    # discard var name and row vals at the start of each line and any extra 
    # data at their end
    data = [[eval(val) for val in line[start_col:start_col+2*ncateg2:2] if val] 
            for line in lines[:ncateg1]]
    return (var1, var2), (colvals, rowvals), data

def save_csv_align(output_path, variables, possible_values, data):
    
    colvals, rowvals = possible_values
    with open(output_path, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow([rename_var(name) for name in variables])
        writer.writerow([''] + colvals)
        for rowval, line in izip(rowvals, data):
            writer.writerow([rowval] + line)

def convert_txt_align(input_path, output_path=None, invert=False):
    if output_path is None:
        # al_regr_p_alive_f.txt -> al_p_alive_f.csv
        fpath, fname = path.split(input_path)
        basename, ext = path.splitext(fname)
        assert basename.startswith('al_regr_')
        output_path = path.join(fpath, "al_%s.csv" % basename[8:])
    print "converting '%s' to '%s'" % (input_path, output_path)
    try:
        variables, possible_values, data = load_txt_align(input_path, invert)
        save_csv_align(output_path, variables, possible_values, data)
    except Exception:
        print "FAILED"
    
if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        print "Usage: %s [input_path] [output_path] [invert]" % args[0]
        sys.exit()
    else:
        input_path = args[1]

    if len(args) < 3:
        output_path = None
    else: 
        output_path = args[2]

    invert = len(args) >= 4 and args[3] == 'invert'
    
    convert_txt_align(input_path, output_path, invert)