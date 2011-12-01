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
  action: can be either 'import' or 'run'
  file: the file to run or import
  -i: show the interactive console after the simulation 
""" % args[0]

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
        simulation = Simulation.from_yaml(fpath)
    
        do_profile = False
        if do_profile:
            import cProfile as profile
            profile.run('simulation.run()', 'c:\\tmp\\simulation.profile')
            # to use profiling data:
            # p = pstats.Stats('c:\\tmp\\simulation.profile')
            # p.strip_dirs().sort_stats('cum').print_stats(30)
        else:
#            try:
            simulation.run(console)
#            except Exception, e:
#                print 
#                print str(e)
#                import traceback
#                with file('error.log', 'w') as f:
#                    traceback.print_exc(file=f)

    elif action == "import":
        try:
            csv2h5(fpath)
        except yaml.parser.ParserError, e:
#            e.context      # while parsing a block mapping
#            e.context_mark # in "import.yml", line 18, column 9
#            e.problem     # expected <block end>, 
                                        # but found '<block sequence start>'
#            e.problem_mark # in "import.yml", line 29, column 12
#            m = e.problem_mark
            print "SYNTAX ERROR %s" % str(e.problem_mark).strip() 
        except yaml.scanner.ScannerError, e:
            print "SYNTAX ERROR %s %s" % (str(e.problem),
                                          str(e.context_mark).strip()) 
        except SyntaxError, e:
            print "SYNTAX ERROR:", str(e)
        except Exception, e:
            print "ERROR:", str(e)
            raise
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
