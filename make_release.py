#!/usr/bin/python
# coding: utf-8
# Release script for LIAM2
# Licence: GPLv3
# Requires:
# * git, pscp and outlook in PATH
# * all tools used for building the doc & exe in PATH
# * website directory in ../liam2-website
from __future__ import print_function

import errno
import fnmatch
import os
import re
import stat
import subprocess
import sys
# import tempfile
import urllib
import zipfile

from datetime import date
from os import chdir, makedirs
from os.path import exists, abspath, dirname
from shutil import copytree, copy2, rmtree as _rmtree
from subprocess import check_output, STDOUT, CalledProcessError

from liam2.compat import input

WEBSITE = 'liam2.plan.be'
TMP_PATH = r"c:\tmp\liam2_new_release"
# not using tempfile.mkdtemp to be able to resume an aborted release
# TMP_PATH = os.path.join(tempfile.gettempdir(), "liam2_new_release")

# TODO:
# - different announce message for pre-releases
# - announce RC on the website too
# ? create a download page for the rc
# - create a conda environment to store requirements for the release
#   create -n liam2-{release} --clone liam2
#   or better yet, only store package versions:
#   conda env export > doc\bundle_environment.yml

# TODO: add more scripts to implement the "git flow" model
# - hotfix_branch
# - release_branch
# - feature_branch
# - make_release, detects hotfix or release


# ------------- #
# generic tools #
# ------------- #

def size2str(value):
    unit = "bytes"
    if value > 1024.0:
        value /= 1024.0
        unit = "Kb"
        if value > 1024.0:
            value /= 1024.0
            unit = "Mb"
        return "%.2f %s" % (value, unit)
    else:
        return "%d %s" % (value, unit)


def generate(fname, **kwargs):
    with open('%s.tmpl' % fname) as in_f, open(fname, 'w') as out_f:
        out_f.write(in_f.read().format(**kwargs))


def _remove_readonly(function, path, excinfo):
    if function in (os.rmdir, os.remove) and excinfo[1].errno == errno.EACCES:
        # add write permission to owner
        os.chmod(path, stat.S_IWUSR)
        # retry removing
        function(path)
    else:
        raise


def rmtree(path):
    _rmtree(path, onerror=_remove_readonly)


def call(*args, **kwargs):
    try:
        return check_output(*args, stderr=STDOUT, **kwargs)
    except CalledProcessError as e:
        print(e.output)
        raise e


def echocall(*args, **kwargs):
    print(' '.join(args))
    return call(*args, **kwargs)


def git_remote_last_rev(url, branch=None):
    """
    :param url: url of the remote repository
    :param branch: an optional branch (defaults to 'refs/heads/master')
    :return: name/hash of the last revision
    """
    if branch is None:
        branch = 'refs/heads/master'
    output = call('git ls-remote %s %s' % (url, branch))
    for line in output.splitlines():
        if line.endswith(branch):
            return line.split()[0]
    raise Exception("Could not determine revision number")


def branchname(statusline):
    """
    computes the branch name from a "git status -b -s" line
    ## master...origin/master
    """
    statusline = statusline.replace('#', '').strip()
    pos = statusline.find('...')
    return statusline[:pos] if pos != -1 else statusline


def yes(msg, default='y'):
    choices = ' (%s/%s) ' % tuple(c.capitalize() if c == default else c
                                  for c in ('y', 'n'))
    answer = None
    while answer not in ('', 'y', 'n'):
        if answer is not None:
            print("answer should be 'y', 'n', or <return>")
        answer = input(msg + choices).lower()
    return (default if answer == '' else answer) == 'y'


def no(msg, default='n'):
    return not yes(msg, default)


def do(description, func, *args, **kwargs):
    print(description + '...', end=' ')
    func(*args, **kwargs)
    print("done.")


def allfiles(pattern, path='.'):
    """
    like glob.glob(pattern) but also include files in subdirectories
    """
    return (os.path.join(dirpath, f)
            for dirpath, dirnames, files in os.walk(path)
            for f in fnmatch.filter(files, pattern))


def zip_pack(archivefname, filepattern):
    with zipfile.ZipFile(archivefname, 'w', zipfile.ZIP_DEFLATED) as f:
        for fname in allfiles(filepattern):
            f.write(fname)


def zip_unpack(archivefname, dest=None):
    with zipfile.ZipFile(archivefname) as f:
        f.extractall(dest)


def short(release_name):
    return release_name[:-2] if release_name.endswith('.0') else release_name


def long_release_name(release_name):
    """
    transforms a short release name such as 0.8 to a long one such as 0.8.0

    >>> long_release_name('0.8')
    '0.8.0'
    >>> long_release_name('0.8.0')
    '0.8.0'
    >>> long_release_name('0.8rc1')
    '0.8.0rc1'
    >>> long_release_name('0.8.0rc1')
    '0.8.0rc1'
    """
    dotcount = release_name.count('.')
    if dotcount >= 2:
        return release_name
    assert dotcount == 1, "%s contains %d dots" % (release_name, dotcount)
    pos = pretag_pos(release_name)
    if pos is not None:
        return release_name[:pos] + '.0' + release_name[pos:]
    return release_name + '.0'


def pretag_pos(release_name):
    """
    gives the position of any pre-release tag
    >>> pretag_pos('0.8')
    >>> pretag_pos('0.8alpha25')
    3
    >>> pretag_pos('0.8.1rc1')
    5
    """
    # 'a' needs to be searched for after 'beta'
    for tag in ('rc', 'c', 'beta', 'b', 'alpha', 'a'):
        match = re.search(tag + r'\d+', release_name)
        if match is not None:
            return match.start()
    return None


def strip_pretags(release_name):
    """
    removes pre-release tags from a version string

    >>> strip_pretags('0.8')
    '0.8'
    >>> strip_pretags('0.8alpha25')
    '0.8'
    >>> strip_pretags('0.8.1rc1')
    '0.8.1'
    """
    pos = pretag_pos(release_name)
    return release_name[:pos] if pos is not None else release_name


def isprerelease(release_name):
    """
    tests whether the release name contains any pre-release tag

    >>> isprerelease('0.8')
    False
    >>> isprerelease('0.8alpha25')
    True
    >>> isprerelease('0.8.1rc1')
    True
    """
    return pretag_pos(release_name) is not None


# -------------------- #
# end of generic tools #
# -------------------- #

# ------------------------- #
# specific helper functions #
# ------------------------- #


def rst2txt(s):
    """
    translates rst to raw text

    >>> rst2txt(":ref:`matching() <matching>`")
    'matching()'
    >>> # \\n needs to be escaped because we are in a docstring
    >>> rst2txt(":ref:`matching()\\n  <matching>`")
    'matching()\\n  '
    >>> rst2txt(":PR:`123`")
    'pull request 123'
    >>> rst2txt(":pr:`123`")
    'pull request 123'
    >>> rst2txt(":issue:`123`")
    'issue 123'
    >>> rst2txt("::")
    ''
    """
    s = s.replace("::", "")
    # first replace :ref:s which span across two lines (we want to *keep* the
    # blanks in those) then those on one line (where we kill the spaces)
    s = re.sub(r":ref:`(.+ *[\n\r] *)<.+>`", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r":ref:`(.+) +<.+>`", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r":pr:`(\d+)`", r"pull request \1", s, flags=re.IGNORECASE)
    return re.sub(r":issue:`(\d+)`", r"issue \1", s, flags=re.IGNORECASE)


def relname2fname(release_name):
    short_version = short(strip_pretags(release_name))
    return r"version_%s.rst.inc" % short_version.replace('.', '_')


def release_changes(context):
    directory = r"doc\usersguide\source\changes"
    fname = relname2fname(context['release_name'])
    with open(os.path.join(context['build_dir'], directory, fname)) as f:
        return f.read().decode('utf-8-sig')


def update_versions(release_name):
    # git clone + install will fail sauf si post-release (mais meme dans ce
    # cas là, ce ne sera pas précis)
    #
    # version in archive I do with make_release: OK

    # doc\usersguide\source\conf.py
    # liam2\setup.py
    # liam2\main.py
    pass


def test_executable(relpath):
    """
    test an executable with relative path *relpath*
    """
    print()
    makedirs('testoutput')
    outpath = abspath('testoutput')
    runcmd = relpath + '/main --output-path ' + outpath + ' run '
    importcmd = relpath + '/main import '
    testspath = 'liam2/tests/functional/'
    demospath = 'liam2/tests/examples/'
    echocall(runcmd + testspath + 'static.yml')
    echocall(runcmd + testspath + 'generate.yml')
    echocall(importcmd + testspath + 'import.yml')
    echocall(runcmd + testspath + 'simulation.yml')
    echocall(runcmd + testspath + 'variant.yml')
    echocall(runcmd + testspath + 'matching.yml')
    echocall(runcmd + demospath + 'demo01.yml')
    echocall(importcmd + demospath + 'demo_import.yml')
    echocall(runcmd + demospath + 'demo01.yml')
    echocall(runcmd + demospath + 'demo02.yml')
    echocall(runcmd + demospath + 'demo03.yml')
    echocall(runcmd + demospath + 'demo04.yml')
    echocall(runcmd + demospath + 'demo05.yml')
    echocall(runcmd + demospath + 'demo06.yml')
    echocall(runcmd + demospath + 'demo07.yml')
    echocall(runcmd + demospath + 'demo08.yml')
    echocall(runcmd + demospath + 'demo09.yml')
    echocall(runcmd + demospath + 'demo10.yml')
    rmtree('testoutput')


def create_source_archive(release_name, rev):
    call(r'git archive --format zip --output ..\LIAM2-%s-src.zip %s'
         % (release_name, rev))


def copy_release(release_name):
    copytree(r'build\bundle\editor', r'win32\editor')
    copytree(r'build\bundle\editor', r'win64\editor')
    copytree(r'build\liam2\tests\examples', r'win32\examples')
    copytree(r'build\liam2\tests\examples', r'win64\examples')
    copytree(r'build\build\exe.win32-2.7', r'win32\liam2')
    copytree(r'build\build\exe.win-amd64-2.7', r'win64\liam2')
    makedirs(r'win32\documentation')
    makedirs(r'win64\documentation')
    copy2(r'build\doc\usersguide\build\htmlhelp\LIAM2UserGuide.chm',
          r'win32\documentation\LIAM2UserGuide.chm')
    copy2(r'build\doc\usersguide\build\htmlhelp\LIAM2UserGuide.chm',
          r'win64\documentation\LIAM2UserGuide.chm')
    # stuff not in the bundles
    copy2(r'build\doc\usersguide\build\latex\LIAM2UserGuide.pdf',
          r'LIAM2UserGuide-%s.pdf' % release_name)
    copy2(r'build\doc\usersguide\build\htmlhelp\LIAM2UserGuide.chm',
          r'LIAM2UserGuide-%s.chm' % release_name)
    copytree(r'build\doc\usersguide\build\html', 'htmldoc')
    copytree(r'build\doc\usersguide\build\web',
             r'webdoc\%s' % short(release_name))


def create_bundle_archives(release_name):
    chdir('win32')
    zip_pack(r'..\LIAM2Suite-%s-win32.zip' % release_name, '*')
    chdir('..')
    chdir('win64')
    zip_pack(r'..\LIAM2Suite-%s-win64.zip' % release_name, '*')
    chdir('..')
    chdir('htmldoc')
    zip_pack(r'..\LIAM2UserGuide-%s-html.zip' % release_name, '*')
    chdir('..')


def check_bundle_archives(release_name):
    """
    checks the bundles unpack correctly
    """
    makedirs('test')
    zip_unpack('LIAM2Suite-%s-win32.zip' % release_name, r'test\win32')
    zip_unpack('LIAM2Suite-%s-win64.zip' % release_name, r'test\win64')
    zip_unpack('LIAM2UserGuide-%s-html.zip' % release_name, r'test\htmldoc')
    zip_unpack('LIAM2-%s-src.zip' % release_name, r'test\src')
    rmtree('test')

# -------------------------------- #
# end of specific helper functions #
# -------------------------------- #


# ----- #
# steps #
# ----- #

def check_local_repo(context):
    # releasing from the local clone has the advantage I can prepare the
    # release offline and only push and upload it when I get back online
    branch, release_name = context['branch'], context['release_name']
    repository, rev = context['repository'], context['rev']

    s = "Using local repository at: %s !" % repository
    print("\n", s, "\n", "=" * len(s), "\n", sep='')

    status = call('git status -s -b')
    lines = status.splitlines()
    statusline, lines = lines[0], lines[1:]
    curbranch = branchname(statusline)
    if curbranch != branch:
        print("%s is not the current branch (%s). "
              "Please use 'git checkout %s'." % (branch, curbranch, branch))
        exit(1)

    if lines:
        uncommited = sum(1 for line in lines if line[1] in 'MDAU')
        untracked = sum(1 for line in lines if line.startswith('??'))
        print('Warning: there are %d files with uncommitted changes '
              'and %d untracked files:' % (uncommited, untracked))
        print('\n'.join(lines))
        if no('Do you want to continue?'):
            exit(1)

    ahead = call('git log --format=format:%%H origin/%s..%s' % (branch, branch))
    num_ahead = len(ahead.splitlines())
    print("Branch '%s' is %d commits ahead of 'origin/%s'"
          % (branch, num_ahead, branch), end='')
    if num_ahead:
        if yes(', do you want to push?'):
            do('Pushing changes', call, 'git push')
    else:
        print()

    if no('Release version %s (%s)?' % (release_name, rev)):
        exit(1)


def create_tmp_directory(context):
    tmp_dir = context['tmp_dir']
    if exists(tmp_dir):
        rmtree(tmp_dir)
    makedirs(tmp_dir)


def clone_repository(context):
    chdir(context['tmp_dir'])

    # make a temporary clone in /tmp. The goal is to make sure we do not
    # include extra/unversioned files. For the -src archive, I don't think
    # there is a risk given that we do it via git, but the risk is there for
    # the bundles (src/build is not always clean, examples, editor, ...)

    # Since this script updates files (update_changelog and build_website), we
    # need to get those changes propagated to GitHub. I do that by updating the
    # temporary clone then push twice: first from the temporary clone to the
    # "working copy clone" (eg ~/devel/liam2) then to GitHub from there. The
    # alternative to modify the "working copy clone" directly is worse because
    # it needs more complicated path handling that the 2 push approach.
    do('Cloning repository', call,
       'git clone -b {branch} {repository} build'.format(**context))


def check_clone(context):
    chdir(context['build_dir'])

    # check last commit
    print()
    print(call('git log -1').decode('utf8', 'replace'))
    print()

    if no('Does that last commit look right?'):
        exit(1)

    if context['public_release']:
        # check release changes
        print(release_changes(context))
        if no('Does the release changelog look right?'):
            exit(1)


def build_exe(context):
    chdir(context['build_dir'])

    context['test_release'] = True if context['public_release'] \
        else yes('Do you want to test the executables after they are created?')

    call('buildall_exe.bat')


def test_executables(context):
    chdir(context['build_dir'])

    if not context.get('test_release', True):
        return

    for arch in ('win32', 'win-amd64'):
        test_executable(r'build\exe.%s-2.7' % arch)


def update_changelog(context):
    """
    Update release date in changes.rst
    """
    chdir(context['build_dir'])

    if not context['public_release']:
        return

    release_name = context['release_name']
    fpath = r'doc\usersguide\source\changes.rst'
    with open(fpath) as f:
        lines = f.readlines()
        title = "Version %s" % short(release_name)
        if lines[5] != title + '\n':
            print("changes.rst not modified (the last release is not %s)"
                  % title)
            return
        release_date = lines[8]
        if release_date != "In development.\n":
            print('changes.rst not modified (the last release date is "%s" '
                  'instead of "In development.", was it already released?)'
                  % release_date)
            return
        lines[8] = "Released on {}.\n".format(date.today().isoformat())
    with open(fpath, 'w') as f:
        f.writelines(lines)
    with open(fpath) as f:
        print('\n'.join(f.read().decode('utf-8-sig').splitlines()[:20]))
    if no('Does the full changelog look right?'):
        exit(1)
    call('git commit -m "update release date in changes.rst" %s' % fpath)


def build_doc(context):
    chdir(context['build_dir'])
    chdir('doc')
    call('buildall.bat')


def create_archives(context):
    chdir(context['build_dir'])

    release_name = context['release_name']
    create_source_archive(release_name, context['rev'])

    chdir(context['tmp_dir'])

    copy_release(release_name)
    create_bundle_archives(release_name)
    check_bundle_archives(release_name)


def final_confirmation(context):
    if not context['public_release']:
        return

    msg = """Is the release looking good? If so, the tag will be created and
pushed, everything will be uploaded to the production server. Stuff to watch
out for:
* version numbers (executable & doc first page & changelog)
* ...
"""
    if no(msg):
        exit(1)


def tag_release(context):
    chdir(context['build_dir'])

    if not context['public_release']:
        return

    release_name = context['release_name']
    call('git tag -a {name} -m "tag release {name}"'.format(name=release_name))


def upload(context):
    chdir(context['tmp_dir'])

    if not context['public_release']:
        return

    release_name = context['release_name']

    # pscp is the scp provided in PuTTY's installer
    base_url = '%s@%s:%s' % ('cic', WEBSITE, WEBSITE)
    # 1) archives
    subprocess.call(r'pscp * %s/download' % base_url)

    # 2) documentation
    chdir('webdoc')
    subprocess.call(r'pscp -r %s %s/documentation' % (short(release_name),
                                                      base_url))


def pull(context):
    if not context['public_release']:
        return

    # pull the changelog commits to the branch (usually master)
    # and the release tag (which refers to the last commit)
    chdir(context['repository'])
    do('Pulling changes in {repository}'.format(**context),
       call, 'git pull --ff-only --tags {build_dir} {branch}'.format(**context))


def push(context):
    if not context['public_release']:
        return

    chdir(context['repository'])
    do('Pushing main repository changes to GitHub',
       call, 'git push origin {branch} --follow-tags'.format(**context))


def cleanup(context):
    chdir(context['tmp_dir'])
    rmtree('win32')
    rmtree('win64')
    # build is needed by the website script
#    rmtree('build')

# ------------ #
# end of steps #
# ------------ #

steps_funcs = [
    (check_local_repo, ''),
    (create_tmp_directory, ''),
    (clone_repository, ''),
    (check_clone, ''),
    (build_exe, 'Building executables'),
    (test_executables, 'Testing executables'),
    (update_changelog, 'Updating changelog'),
    (build_doc, 'Building doc'),
    (create_archives, 'Creating archives'),
    (final_confirmation, ''),
    (tag_release, 'Tagging release'),
    # We used to push from /tmp to the local repository but you cannot push
    # to the currently checked out branch of a repository, so we need to
    # pull changes instead. However pull (or merge) add changes to the
    # current branch, hence we make sure at the beginning of the script
    # that the current git branch is the branch to release. It would be
    # possible to do so without a checkout by using:
    # git fetch {tmp_path} {branch}:{branch}
    # instead but then it only works for fast-forward and non-conflicting
    # changes. So if the working copy is dirty, you are out of luck.
    (pull, ''),
    # >>> need internet from here
    (push, ''),
    (upload, 'Uploading'),
    (cleanup, 'Cleaning up')
]


def make_release(release_name='dev', steps=':', branch='master'):
    func_names = [f.__name__ for f, desc in steps_funcs]
    if ':' in steps:
        start, stop = steps.split(':')
        start = func_names.index(start) if start else 0
        # + 1 so that stop bound is inclusive
        stop = func_names.index(stop) + 1 if stop else len(func_names)
    else:
        # assuming a single step
        start = func_names.index(steps)
        stop = start + 1

    if release_name != 'dev':
        if 'pre' in release_name:
            raise ValueError("'pre' is not supported anymore, use 'alpha' or "
                             "'beta' instead")
        if '-' in release_name:
            raise ValueError("- is not supported anymore")

        release_name = long_release_name(release_name)

    repository = abspath(dirname(__file__))
    rev = git_remote_last_rev(repository, 'refs/heads/%s' % branch)
    public_release = release_name != 'dev'
    if not public_release:
        # take first 7 digits of commit hash
        release_name = rev[:7]

    context = {'branch': branch, 'release_name': release_name, 'rev': rev,
               'repository': repository,
               'tmp_dir': TMP_PATH,
               'build_dir': os.path.join(TMP_PATH, 'build'),
               'public_release': public_release}
    for step_func, step_desc in steps_funcs[start:stop]:
        if step_desc:
            do(step_desc, step_func, context)
        else:
            step_func(context)

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) < 2:
        print("Usage: %s release_name|dev [step|startstep:stopstep] [branch]"
              % argv[0])
        print("steps:", ', '.join(f.__name__ for f, _ in steps_funcs))
        sys.exit()

    make_release(*argv[1:])
