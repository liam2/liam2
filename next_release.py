#!/usr/bin/python
# coding=utf-8
# script to start a new release cycle
# Licence: GPLv3
from os.path import join
from make_release import relname2fname


def add_release(release_name):
    # create "empty" change file for that release
    fname = relname2fname(release_name)
    with open(r'doc\usersguide\source\changes\template.rst.inc') as f:
        changes_template = f.read()
    with open(join(r'doc\usersguide\source\changes', fname), 'w') as f:
        f.write(changes_template)


if __name__ == '__main__':
    from sys import argv

    add_release(*argv[1:])
