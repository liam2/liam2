from simulation import Simulation
from data_main import csv2h5

__version__ = "0.3.1dev"

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
    console = len(args) > 3 and args[3] == "-i"
    
    if action == 'run':
        print "Using simulation file: '%s'" % fpath
        simulation = Simulation(fpath, console)
    
        do_profile = False
        if do_profile:
            import cProfile as profile
            profile.run('simulation.run()', 'c:\\tmp\\simulation.profile')
            # to use profiling data:
            # p = pstats.Stats('c:\\tmp\\simulation.profile')
            # p.strip_dirs().sort_stats('cum').print_stats(30)
        else:
#            try:
            simulation.run()
#            except Exception, e:
#                print 
#                print str(e)
#                import traceback
#                with file('error.log', 'w') as f:
#                    traceback.print_exc(file=f)

    elif action == "import":
        csv2h5(fpath)
    else:
        usage(args)
