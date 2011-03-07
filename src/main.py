from simulation import Simulation

if __name__ == '__main__':
    import sys

    print 'Using Python %s' % sys.version
    args = sys.argv
    fpath = args[1] if len(args) > 1 else 'simulation.yaml'
    console = len(args) > 2 and args[2] == "-i" 
    print "using simulation file: '%s'" % fpath
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
    
