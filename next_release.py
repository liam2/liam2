#!/usr/bin/python
# encoding: utf-8
# script to start a new release cycle
# Licence: GPLv3
from os.path import join
from shutil import copy

from make_release import relname2fname, short, long_release_name


def add_release(release_name):
    release_name = long_release_name(release_name)
    fname = relname2fname(release_name)

    # create "empty" changelog for that release
    changes_dir = r'doc\usersguide\source\changes'
    copy(join(changes_dir, 'template.rst.inc'),
         join(changes_dir, fname))

    # include release changelog in changes.rst
    fpath = r'doc\usersguide\source\changes.rst'
    with open(fpath) as f:
        lines = f.readlines()
        title = f"Version {short(release_name)}"
        if lines[5] == title + '\n':
            print(f"changes.rst not modified (it already contains {title})")
            return
        this_version = f"""{title}
{"=" * len(title)}

In development.

.. include:: {'changes/' + fname}
   
    
"""
        lines[5:5] = this_version.splitlines(True)
    with open(fpath, 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    from sys import argv

    add_release(*argv[1:])
