import subprocess

def get_last_rev(url):
    output = subprocess.check_output('svn info %s' % url)
    lines = output.splitlines()
    for line in lines:
        if line.startswith('Revision'):
            return int(line[10:])
    raise Exception("Could not determine revision number")

if __name__=='__main__':
    from sys import argv, exit
    
    if len(argv) < 2:
        print "Usage: %s repository" % (argv[0],)
        exit(2)

    print get_last_rev(argv[1])