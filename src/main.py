from simulation import Simulation

__version__ = "0.2.1"

class AutoflushFile(object):
    def __init__(self, f):
        self.f = f

    def write(self, s):
        self.f.write(s)
        self.f.flush()

if __name__ == '__main__':
    import sys, platform

    sys.stdout = AutoflushFile(sys.stdout)
    sys.stderr = AutoflushFile(sys.stderr)
    print "LIAM2 %s using Python %s (%s)" % (__version__, 
                                             platform.python_version(),
                                             platform.architecture()[0])
    print

    args = sys.argv
    fpath = args[1] if len(args) > 1 else 'simulation.yml'
    console = len(args) > 2 and args[2] == "-i" 
    print "Using simulation file: '%s'" % fpath
    simulation = Simulation(fpath, console)
    
    do_profile = False
    if do_profile:
        import cProfile as profile
        profile.run('simulation.run()', 'c:\\tmp\\simulation.profile')
    else:
#        try:
        simulation.run()
#        except Exception, e:
#            print 
#            print str(e)
#            import traceback
#            with file('error.log', 'w') as f:
#                traceback.print_exc(file=f)

    # use profiling data:
    # p = pstats.Stats('c:\\tmp\\simulation.profile')
    # p.strip_dirs().sort_stats('cum').print_stats(30)
    
