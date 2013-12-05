#!/usr/bin/python
# Release script for Liam2
# Licence: GPLv3
from __future__ import print_function

import errno
import os
import stat

from os import chdir, makedirs
from os.path import exists
from shutil import copytree, copy2, rmtree as _rmtree
from subprocess import check_output as call


def shell(*args):
    return call(*args, shell=True)


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


def make_release(release_name=None, branch=None):
    # Since git is a dvcs, we could make this script work locally, but it would
    # not be any more useful because making a release is usually to distribute
    # it to someone, and for that I need network access anyway.
    # Furthermore, cloning from the remote repository makes sure we do not
    # include any untracked file

    # git config --get remote.origin.url
    repository = 'https://github.com/liam2/liam2.git'
    if branch is None:
        branch = 'master'

    status = call('git status -s')
    lines = status.splitlines()
    if lines:
        uncommited = sum(1 for line in lines if line.startswith(' M'))
        untracked = sum(1 for line in lines if line.startswith('??'))
        print('Warning: there are %d files with uncommitted changes '
              'and %d untracked files:' % (uncommited, untracked))
        print(status)
        yn = raw_input('Do you want to abort? (Y/n)')
        if not yn.lower().startswith('n'):
            exit(1)

    ahead = call('git log --format=format:%%H origin/%s..%s' % (branch, branch))
    num_ahead = len(ahead.splitlines())
    print('%s is %d commits ahead of origin/%s' % (branch, num_ahead, branch),
          end='')
    if num_ahead:
        yn = raw_input(', do you want to push? (Y/n)')
        if not yn.lower().startswith('n'):
            print('pushing changes...')
            call('git push')
            print('done.')
    else:
        print()

    rev = git_remote_last_rev(repository, 'refs/heads/%s' % branch)

    if release_name is None:
        # take first 7 digits of commit hash
        release_name = rev[:7]

    yn = raw_input("Release version %s (%s)? (y/N)" % (release_name, rev))
    if not yn.lower().startswith('y'):
        exit(1)

    chdir('c:\\tmp')
    if exists('liam2_new_release'):
        rmtree('liam2_new_release')
    makedirs('liam2_new_release')
    chdir('liam2_new_release')

    print()
    call('git clone -b %s %s build' % (branch, repository))
    print()

    chdir('build')

    print()
    print(call('git log -1').decode('utf8'))
    print()

    yn = raw_input("does that last commit look right? (y/N)")
    if not yn.lower().startswith('y'):
        exit(1)

    call('git archive --format zip --output liam2-%s-src.zip %s'
         % (release_name, rev))
    call('buildall.bat')

    print("moving stuff around...")
    chdir('..')

    copytree('build\\bundle\\editor', 'win32\\editor')
    copytree('build\\bundle\\editor', 'win64\\editor')

    copytree('build\\tests\\examples', 'win32\\examples')
    copytree('build\\tests\\examples', 'win64\\examples')

    copytree('build\\src\\build\\exe.win32-2.7', 'win32\\liam2')
    copytree('build\\src\\build\\exe.win-amd64-2.7', 'win64\\liam2')

    copytree('build\\doc\\usersguide\\build\\html',
             'win32\\documentation\\html')
    copytree('build\\doc\\usersguide\\build\\html',
             'win64\\documentation\\html')
    copytree('build\\doc\\usersguide\\build\\html', 'html\\%s' % release_name)

    copy2('build\doc\usersguide\build\LIAM2UserGuide.pdf',
          'win32\documentation')
    copy2('build\doc\usersguide\build\LIAM2UserGuide.pdf',
          'win64\documentation')
    copy2('build\doc\usersguide\build\LIAM2UserGuide.pdf',
          'LIAM2UserGuide-%s.pdf' % release_name)
    copy2('build\doc\usersguide\build\LIAM2UserGuide.chm',
          'LIAM2UserGuide-%s.chm' % release_name)
    print("done.")

    chdir('win32')
    call('7z a -tzip ..\Liam2Suite-%s-win32.zip *' % release_name)
    chdir('..')
    chdir('win64')
    call('7z a -tzip ..\Liam2Suite-%s-win64.zip *' % release_name)
    chdir('..')

    rmtree('win32')
    rmtree('win64')

    if release_name != rev[:7]:
        yn = raw_input("is the release looking good (if so, the tag will be "
                       "created and pushed)? (y/N)")
        if not yn.lower().startswith('y'):
            exit(1)

        print("tagging release...", end=' ')
        call('git tag -a %{name}s -m "tag release %{name}s"'
             % {'name': release_name})
        print("done")

        print('pushing tag...', end=' ')
        call('git push')
        print("done")

    rmtree('build')


if __name__=='__main__':
    from sys import argv

    # if len(argv) < 2:
    #     print "Usage: %s [version] [branch]" % (argv[0],)
    #     exit(2)

    make_release(*argv[1:])

