import subprocess


def get_svn_last_rev(url):
    """
    url of the remote repository
    """
    output = subprocess.check_output('svn info %s' % url)
    lines = output.splitlines()
    for line in lines:
        if line.startswith('Revision'):
            return int(line[10:])
    raise Exception("Could not determine revision number")


def get_git_local_last_rev(path):
    """
    path is the path to the repository without trailing slash
    """
    output = subprocess.check_output('git --git-dir %s\.git log -n 1' % path)
    lines = output.splitlines()
    for line in lines:
        if line.startswith('commit'):
            x = line[6:]
            print 'rev:', repr(x)
            return x
            # return line[6:]
    raise Exception("Could not determine revision number")


def get_git_remote_last_rev(url, branch='refs/heads/master'):
    """
    url of the remote repository
    branch is an optional branch (defaults to 'refs/heads/master')
    """
    output = subprocess.check_output('git ls-remote %s %s' % (url, branch))
    lines = output.splitlines()
    for line in lines:
        if line.endswith(branch):
            return line.split()[0]
    raise Exception("Could not determine revision number")

if __name__ == '__main__':
    from sys import argv, exit
    
    if len(argv) < 2:
        print "Usage: %s repository" % (argv[0],)
        exit(2)

    print get_git_remote_last_rev(argv[1])