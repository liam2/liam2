from simulation import Simulation

__version__ = "0.2dev"

class AutoflushFile(object):
    def __init__(self, f):
        self.f = f

    def write(self, s):
        self.f.write(s)
        self.f.flush()

if __name__ == '__main__':
    import sys, platform

    sys.stdout = AutoflushFile(sys.stdout)
    print "LIAM2 %s using Python %s (%s)\n" % (__version__, 
                                               platform.python_version(),
                                               platform.architecture()[0])
                                  
    args = sys.argv
    fpath = args[1] if len(args) > 1 else 'simulation.yaml'
    console = len(args) > 2 and args[2] == "-i" 
    print "Using simulation file: '%s'" % fpath
    simulation = Simulation(fpath, console)
    
    do_profile = False
    if do_profile:
        import cProfile as profile
        profile.run('simulation.run()', 'c:\\tmp\\simulation.profile')
    else:
        simulation.run()

    # use profiling data:
    # p = pstats.Stats('c:\\tmp\\simulation.profile')
    # p.strip_dirs().sort_stats('cum').print_stats(30)
    
