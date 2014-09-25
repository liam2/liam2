#!/usr/bin/python
# coding=utf-8
# Release script for LIAM2
# Licence: GPLv3
from __future__ import print_function

import errno
import fnmatch
import os
import stat
import subprocess
import zipfile
import re

from datetime import date
from os import chdir, makedirs
from os.path import exists, getsize, abspath
from shutil import copytree, copy2, rmtree as _rmtree
from subprocess import check_output, STDOUT, CalledProcessError

WEBSITE = 'liam2.plan.be'

#TODO: add more scripts to implement the "git flow" model
# - hotfix_branch
# - release_branch
# - feature_branch
# - make_release, detects hotfix or release


#---------------#
# generic tools #
#---------------#

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
    except CalledProcessError, e:
        print(e.output)
        raise e


def echocall(*args, **kwargs):
    print(' '.join(args))
    return call(*args, **kwargs)


def git_remote_last_rev(url, branch=None):
    """
    url of the remote repository
    branch is an optional branch (defaults to 'refs/heads/master')
    """
    if branch is None:
        branch = 'refs/heads/master'
    output = call('git ls-remote %s %s' % (url, branch))
    for line in output.splitlines():
        if line.endswith(branch):
            return line.split()[0]
    raise Exception("Could not determine revision number")


def yes(msg, default='y'):
    choices = ' (%s/%s) ' % tuple(c.capitalize() if c == default else c
                                  for c in ('y', 'n'))
    answer = None
    while answer not in ('', 'y', 'n'):
        if answer is not None:
            print("answer should be 'y', 'n', or <return>")
        answer = raw_input(msg + choices).lower()
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


def short(rel_name):
    return rel_name[:-2] if rel_name.endswith('.0') else rel_name


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
    if 'pre' in release_name:
        raise ValueError("'pre' is not supported anymore, use 'alpha' or "
                         "'beta' instead")
    if '-' in release_name:
        raise ValueError("- is not supported anymore")

    # 'a' needs to be searched for after 'beta'
    for tag in ('rc', 'c', 'beta', 'b', 'alpha', 'a'):
        release_name = re.sub(tag + '\d+', '', release_name)
    return release_name


#----------------------#
# end of generic tools #
#----------------------#


changelog_template = """{title}
{underline}

Released on {date}.

.. include:: {fpath}


"""


def update_versions(release_name):
    # git clone + install will fail sauf si post-release (mais meme dans ce
    # cas là, ce ne sera pas précis)
    #
    # version in archive I do with make_release: OK

    # doc\usersguide\source\conf.py
    # src\setup.py
    # src\main.py
    pass


def update_changelog(release_name):
    fname = relname2fname(release_name)

    # include it in changes.rst
    fpath = r'doc\usersguide\source\changes.rst'
    with open(fpath) as f:
        lines = f.readlines()
        title = "Version %s" % short(release_name)
        if lines[5] == title:
            print("changes.rst not modified (it already contains %s)" % title)
            return
        variables = dict(title=title,
                         underline="=" * len(title),
                         date=date.today().isoformat(),
                         fpath='changes/' + fname)
        this_version = changelog_template.format(**variables)
        lines[5:5] = this_version.splitlines(keepends=True)
    with open(fpath, 'w') as f:
        f.writelines(lines)
    call('git commit -m "include release changes (%s) in changes.rst" %s'
         % (fname, fpath))
    do('pushing changes.rst', call, 'git push')


def copy_release(release_name):
    copytree(r'build\bundle\editor', r'win32\editor')
    copytree(r'build\bundle\editor', r'win64\editor')
    copytree(r'build\tests\examples', r'win32\examples')
    copytree(r'build\tests\examples', r'win64\examples')
    copytree(r'build\src\build\exe.win32-2.7', r'win32\liam2')
    copytree(r'build\src\build\exe.win-amd64-2.7', r'win64\liam2')
    makedirs(r'win32\documentation')
    makedirs(r'win64\documentation')
    copy2(r'build\doc\usersguide\build\htmlhelp\LIAM2UserGuide.chm',
          r'win32\documentation\LIAM2UserGuide.chm')
    copy2(r'build\doc\usersguide\build\htmlhelp\LIAM2UserGuide.chm',
          r'win64\documentation\LIAM2UserGuide.chm')
    # standalone docs
    copy2(r'build\doc\usersguide\build\latex\LIAM2UserGuide.pdf',
          r'LIAM2UserGuide-%s.pdf' % release_name)
    copy2(r'build\doc\usersguide\build\htmlhelp\LIAM2UserGuide.chm',
          r'LIAM2UserGuide-%s.chm' % release_name)
    copytree(r'build\doc\usersguide\build\html',
             r'html\%s' % short(release_name))


def create_bundles(release_name):
    chdir('win32')
    zip_pack(r'..\LIAM2Suite-%s-win32.zip' % release_name, '*')
    chdir('..')
    chdir('win64')
    zip_pack(r'..\LIAM2Suite-%s-win64.zip' % release_name, '*')
    chdir('..')
    chdir(r'html\%s' % short(release_name))
    zip_pack(r'..\..\LIAM2UserGuide-%s-html.zip' % release_name, '*')
    chdir(r'..\..')


def test_bundle(archivefname, dest):
    zip_unpack(archivefname, dest)
    # we use --debug so that errorlevel is set
    main_dbg = dest + r'\liam2\main --debug '
    echocall(main_dbg + r'run src\tests\functional\generate.yml')
    echocall(main_dbg + r'import src\tests\functional\import.yml')
    echocall(main_dbg + r'run src\tests\functional\simulation.yml')
    echocall(main_dbg + r'run src\tests\functional\variant.yml')
    try:
        chdir(dest)
        main_dbg = r'liam2\main --debug '
        echocall(main_dbg + r'run examples\demo01.yml')
        echocall(main_dbg + r'import examples\demo_import.yml')
        echocall(main_dbg + r'run examples\demo01.yml')
        echocall(main_dbg + r'run examples\demo02.yml')
        echocall(main_dbg + r'run examples\demo03.yml')
        echocall(main_dbg + r'run examples\demo04.yml')
        echocall(main_dbg + r'run examples\demo05.yml')
        echocall(main_dbg + r'run examples\demo06.yml')
        echocall(main_dbg + r'run examples\demo07.yml')
        echocall(main_dbg + r'run examples\demo08.yml')
        echocall(main_dbg + r'run examples\demo09.yml')
    finally:
        chdir('..')


def test_bundles(release_name):
    makedirs('test')
    chdir('test')
    zip_unpack(r'..\LIAM2-%s-src.zip' % release_name, 'src')
    for arch in ('win32', 'win64'):
        test_bundle(r'..\LIAM2Suite-%s-%s.zip' % (release_name, arch), arch)
    chdir('..')
    rmtree('test')


def build_website(release_name):
    fnames = ["LIAM2Suite-%s-win32.zip", "LIAM2Suite-%s-win64.zip",
              "LIAM2-%s-src.zip"]
    s32b, s64b, ssrc = [size2str(getsize(fname % release_name))
                        for fname in fnames]

    chdir(r'build\doc\website')

    generate(r'conf.py', version=short(release_name))
    generate(r'pages\download.rst',
             version=release_name, short_version=short(release_name),
             size32b=s32b, size64b=s64b, sizesrc=ssrc)
    generate(r'pages\documentation.rst',
             version=release_name, short_version=short(release_name))

    title = 'Version %s released' % short(release_name)
    # strip is important otherwise it contains a \n and git chokes on it
    fname = call('tinker -f -p "%s"' % title).strip()

    call('buildall.bat')

    call('start ' + abspath(r'blog\html\index.html'), shell=True)
    call('start ' + abspath(r'blog\html\pages\download.html'), shell=True)
    call('start ' + abspath(r'blog\html\pages\documentation.html'), shell=True)

    if no('Does the website look good?'):
        exit(1)

    call('git add master.rst')
    call('git add %s' % fname)
    call('git commit -m "announce version %s on website"' % short(release_name))
    do('Git-pushing website', call, 'git push')

    chdir(r'..\..\..')


def upload(release_name):
    # pscp is the scp provided in PuTTY's installer
    base_url = '%s@%s:%s' % ('cic', WEBSITE, WEBSITE)
    # archives
    subprocess.call(r'pscp * %s/download' % base_url)

    # documentation
    chdir(r'html')
    subprocess.call(r'pscp -r %s %s/documentation' % (short(release_name),
                                                      base_url))
    chdir(r'..')

    # website
    chdir(r'build\doc\website\blog\html')
    subprocess.call(r'pscp -r * %s' % base_url)
    chdir(r'..\..\..\..\..')


def announce(release_name, changes):
    body = """\
I am pleased to announce that version %s of LIAM2 is now available.

The highlights of this release are:
- x
- y

More details and the complete list of changes are available below.

This new release can be downloaded on our website:
http://liam2.plan.be/pages/download.html

As always, *any* feedback is very welcome, preferably on the liam2-users
mailing list: liam2-users@googlegroups.com (you need to register to be
able to post).

%s
""" % (short(release_name), changes)
    # preselectid='id1' selects the first "identity" for the "from" field
    # We do not use our usual call because the command returns an exit status
    # of 1 (failure) instead of 0, even if it works, so we simply ignore
    # the failure.
    subprocess.call("thunderbird -compose \""
                    "preselectid='id1',"
                    "to='liam2-announce@googlegroups.com',"
                    "subject='Version %s released',"
                    "body='%s'\"" % (short(release_name), body))


def cleanup():
    rmtree('win32')
    rmtree('win64')
    rmtree('build')


def relname2fname(release_name):
    short_version = short(strip_pretags(release_name))
    return r"version_%s.rst.inc" % short_version.replace('.', '_')


def make_release(release_name=None, branch='master'):
    # Since git is a dvcs, we could make this script work locally, but it would
    # not be any more useful because making a release is usually to distribute
    # it to someone, and for that I need network access anyway.
    # Furthermore, cloning from the remote repository makes sure we do not
    # include any untracked file
    # Note, that it is exactly the same syntax, except that repository is a path
    # instead of an URL 'git clone -b %s %s build' % (branch, repository)
    # the only drawback I see is that I could miss changes from others, but
    # we are not there yet :)

    # git ls-remote does not seem to support user-tagged urls
    # repository = call('git config --get remote.origin.url')
    repository = 'https://github.com/liam2/liam2.git'

    status = call('git status -s')
    lines = status.splitlines()
    if lines:
        uncommited = sum(1 for line in lines if line.startswith(' M'))
        untracked = sum(1 for line in lines if line.startswith('??'))
        print('Warning: there are %d files with uncommitted changes '
              'and %d untracked files:' % (uncommited, untracked))
        print(status)
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

    rev = git_remote_last_rev(repository, 'refs/heads/%s' % branch)

    public_release = release_name is not None
    if release_name is None:
        # take first 7 digits of commit hash
        release_name = rev[:7]

    if no('Release version %s (%s)?' % (release_name, rev)):
        exit(1)

    chdir(r'c:\tmp')
    if exists('liam2_new_release'):
        rmtree('liam2_new_release')
    makedirs('liam2_new_release')
    chdir('liam2_new_release')

    do('Cloning', call, 'git clone -b %s %s build' % (branch, repository))

    # ---------- #
    chdir('build')
    # ---------- #

    print()
    print(call('git log -1').decode('utf8'))
    print()

    if no('Does that last commit look right?'):
        exit(1)

    if public_release:
        test_release = True
        fpath = "doc\usersguide\source\changes\\" + relname2fname(release_name)
        with open(fpath) as f:
            changes = f.read().decode('utf-8-sig')
            print(changes)
        if no('Does this changelog look good?'):
            exit(1)
        update_changelog(release_name)
    else:
        test_release = yes('Do you want to test the bundles after they are '
                           'created?')
        changes = ''

    do('Creating source archive', call,
       r'git archive --format zip --output ..\LIAM2-%s-src.zip %s'
       % (release_name, rev))

    do('Building everything', call, 'buildall.bat')

    # ------- #
    chdir('..')
    # ------- #

    do('Moving stuff around', copy_release, release_name)
    do('Creating bundles', create_bundles, release_name)
    if test_release:
        do('Testing bundles', test_bundles, release_name)

    if public_release:
        do('Building website (news, download and documentation pages)',
           build_website, release_name)

        if no('Is the release looking good? If so, the tag will be created and '
              'pushed, everything will be uploaded to the production server '
              'and the release will be announced.'):
            exit(1)

        # ---------- #
        chdir('build')
        # ---------- #

        do('Tagging release', call,
           'git tag -a %(name)s -m "tag release %(name)s"'
           % {'name': release_name})
        do('Pushing tag', call, 'git push origin %s' % release_name)

        # ------- #
        chdir('..')
        # ------- #

        do('Uploading', upload, release_name)

        do('Announcing', announce, release_name, changes)

    do('Cleaning up', cleanup)


if __name__ == '__main__':
    from sys import argv

    # chdir(r'c:\tmp')
    # chdir('liam2_new_release')
    # test_bundles(*argv[1:])
    # build_website(*argv[1:])
    # upload(*argv[1:])
    # announce(*argv[1:])

    make_release(*argv[1:])
