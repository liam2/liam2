#!/usr/bin/python
# coding=utf-8
# script to start a new release cycle
# Licence: GPLv3
from os.path import join
from make_release import relname2fname, short


def add_release(release_name):
    # create "empty" changelog for that release
    fname = relname2fname(release_name)
    with open(r'doc\usersguide\source\changes\template.rst.inc') as f:
        changes_template = f.read()
    with open(join(r'doc\usersguide\source\changes', fname), 'w') as f:
        f.write(changes_template)

    # include release changelog in changes.rst
    fpath = r'doc\usersguide\source\changes.rst'
    changelog_template = """{title}
{underline}

In development.

.. include:: {fpath}


"""

    with open(fpath) as f:
        lines = f.readlines()
        title = "Version %s" % short(release_name)
        if lines[5] == title + '\n':
            print("changes.rst not modified (it already contains %s)" % title)
            return
        this_version = changelog_template.format(title=title,
                                                 underline="=" * len(title),
                                                 fpath='changes/' + fname)
        lines[5:5] = this_version.splitlines(True)
    with open(fpath, 'w') as f:
        f.writelines(lines)

if __name__ == '__main__':
    from sys import argv

    add_release(*argv[1:])
