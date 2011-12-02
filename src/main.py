from os.path import splitext

import yaml

from simulation import Simulation
from data_main import csv2h5
from console import Console
from data import populate_registry

__version__ = "0.4.1dev"

class AutoflushFile(object):
    def __init__(self, f):
        self.f = f

    def write(self, s):
        self.f.write(s)
        self.f.flush()

def usage(args):
    print """
Usage: %s action file [-i]
  action: can be either 'import', 'run' or 'explore'
  file: the file to run, import or explore
  -i: show the interactive console after the simulation 
""" % args[0]

def eat_traceback(func, *args, **kwargs):
# e.context      | while parsing a block mapping
# e.context_mark | in "import.yml", line 18, column 9
# e.problem      | expected <block end>, but found '<block sequence start>'
# e.problem_mark | in "import.yml", line 29, column 12
    try:
        try:
            return func(*args, **kwargs)
        except Exception, e:
            import traceback
            with file('error.log', 'w') as f:
                traceback.print_exc(file=f)
            raise
    except yaml.parser.ParserError, e:
        # eg, inconsistent spacing, no space after a - in a list, ...
        print "SYNTAX ERROR %s" % str(e.problem_mark).strip() 
    except yaml.scanner.ScannerError, e:
        # eg, tabs, missing colon for mapping. The reported problem is different when
        # it happens on the first line (no context_mark) and when it happens on
        # a subsequent line.
        if e.context_mark is not None:
            if e.problem == "could not found expected ':'":
                msg = "could not find expected ':'"
            else:
                msg = e.problem
            mark = e.context_mark
        else:
            if e.problem == "found character '\\t' that cannot start any token":
                msg = "found a TAB character instead of spaces"
            else:
                msg = ""
            mark = e.problem_mark
        if msg:
            msg = ": " + msg
        print "SYNTAX ERROR %s%s" % (str(mark).strip(), msg)
    except yaml.reader.ReaderError, e:
        if e.encoding == 'utf8':
            print "\nERROR in '%s': invalid character found, this probably " \
                  "means you have used non ASCII characters (accents and " \
                  "other non-english characters) and did not save your file " \
                  "using the UTF8 encoding" % e.name
        else:
            raise
    except SyntaxError, e:
        print "SYNTAX ERROR:", e.msg.replace('EOF', 'end of block')
        if e.text is not None:
            print e.text
            offset_str = ' ' * (e.offset - 1) if e.offset > 0 else '' 
            print offset_str + '^'
    except Exception, e:
        print "\nERROR:", str(e)

if __name__ == '__main__':
    import sys, platform

    sys.stdout = AutoflushFile(sys.stdout)
    sys.stderr = AutoflushFile(sys.stderr)
    print "LIAM2 %s using Python %s (%s)" % (__version__, 
                                             platform.python_version(),
                                             platform.architecture()[0])
    print

    args = sys.argv
    if len(args) < 3:
        usage(args)
        sys.exit()
    
    action = args[1]
    fpath = args[2]
    
    if action == 'run':
        print "Using simulation file: '%s'" % fpath
        console = len(args) > 3 and args[3] == "-i"
        simulation = eat_traceback(Simulation.from_yaml, fpath)
        # if an exception ate the simulation with the traceback ;-)
        if simulation is not None:
            eat_traceback(simulation.run, console)
    
#        import cProfile as profile
#        profile.run('simulation.run()', 'c:\\tmp\\simulation.profile')
        # to use profiling data:
        # p = pstats.Stats('c:\\tmp\\simulation.profile')
        # p.strip_dirs().sort_stats('cum').print_stats(30)
    elif action == "import":
        eat_traceback(csv2h5, fpath)
    elif action == "explore":
        _, ext = splitext(fpath)
        if ext in ('.h5', '.hdf5'):
            ftype = 'data'
            h5in = populate_registry(fpath)
            h5out = None
        else:
            ftype = 'simulation'
            simulation = Simulation.from_yaml(fpath)
            h5in, h5out, periodic_globals = simulation.load()
        try:
            print "Using %s file: '%s'" % (ftype, fpath)
            c = Console()
            c.run()
        finally:
            h5in.close()
            if h5out is not None:
                h5out.close()
    else:
        usage(args)
