# encoding: utf-8
from __future__ import absolute_import, division, print_function

import os.path
import re
from glob import glob

__version__ = '0.3'


def upgrade_str(content):
    # 1) transform *link(linkname, ...) -> linkname.*(...)
    content = re.sub(r"([a-zA-Z]+)link\s*\(\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*,?\s*",
                     r"\2.\1(",
                     content)

    # Even though, min/max(...) can now have two meanings we do not need a
    # special case for them, ie we do not need to explicitly convert:
    # grpmin(expr1, expr2, ...) -> min(expr1, filter=expr2, ...)
    # grpmax(expr1, expr2, ...) -> max(expr1, filter=expr2, ...)
    # because ever since grpmin and grpmax accepted a filter argument, it has
    # always been a keyword only argument. There still might be a problem
    # if people used: grpmin(expr1, 0) where 0 is the axis number but since
    # this is undocumented it is very unlikely to have been used by anyone.

    # 2) grpXYZ(...) -> XYZ(...)
    content = re.sub(r"grp([a-zA-Z]+)\s*\(", r"\1(", content)

    # 3) function: -> function():
    # keepends=True so that we keep the line ending style (windows, unix, ...) intact
    lines = content.splitlines(True)
    cur_indent_len = 0
    last_indent = -1
    parent_per_level = {}
    # - is a valid character in yaml sections (and currently used in models). See issues #270 and #271.
    section_re = re.compile(r'^(\s*)([a-zA-Z_][\w-]*):(\s*(#.*)?$)')
    for i in range(len(lines)):
        line = lines[i]

        # ignore blank or commented lines
        if not line or line.isspace() or re.match(r'^\s*#.*$', line):
            continue
        section_match = section_re.match(line)
        if not section_match:
            continue

        cur_indent = section_match.group(1)
        cur_indent_len = len(cur_indent)
        current_section = section_match.group(2)
        for indent in range(cur_indent_len, last_indent + 1):
            parent_per_level.pop(indent, None)
        parent_per_level[cur_indent_len] = current_section
        current_path = [parent_per_level[indent] for indent in sorted(parent_per_level.keys())
                        if indent < cur_indent_len]
        if len(current_path) == 3 and current_path[0] == 'entities' and current_path[2] == 'processes':
            lines[i] = section_re.sub(r"\1\2():\3", line)
        last_indent = cur_indent_len
    content = ''.join(lines)

    # 4) return result
    return content


def upgrade_one(inpath, outpath=None):
    if outpath is not None:
        print("Upgrading '%s'..." % inpath, end=' ')
    else:
        print("* '%s'..." % inpath, end=' ')

    if outpath is None:
        outpath = inpath

    # read original
    with open(inpath, "rb") as f:
        content = f.read()

    # do this before opening the destination file for write to avoid writing a blank file in case the
    # update_str fails for some reason
    updated_content = upgrade_str(content)
    if updated_content != content:
        # make a copy, if needed
        if outpath == inpath:
            filename, _ = os.path.splitext(inpath)
            backup_path = filename + ".bak"
            i = 2
            while os.path.exists(backup_path):
                backup_path = filename + ".bak{}".format(i)
                i += 1
            # print("original model copied to: '%s'" % backup_path)
            with open(backup_path, "wb") as f:
                f.write(content)
        else:
            backup_path = None

        # writing back modified content
        with open(outpath, "wb") as f:
            f.write(updated_content)

        if backup_path is not None:
            print("done (original copied to '%s')" % backup_path)
        else:
            print("done (written to '%s')" % outpath)
    else:
        print("skipped (nothing to update)")


def upgrade(pattern, outpath=None):
    if os.path.isdir(pattern):
        pattern = os.path.join(pattern, '*.yml')
    fnames = glob(pattern)
    if not fnames:
        raise ValueError("No file found matching: {}".format(os.path.abspath(pattern)))

    if len(fnames) > 1 and outpath is not None:
        raise ValueError("Cannot specify output path when using multiple input files")
    for fname in fnames:
        upgrade_one(fname, outpath)


if __name__ == '__main__':
    import sys

    print("LIAM2 model upgrader %s\n" % __version__)

    args = sys.argv
    if len(args) < 2:
        print("""\
Usage
=====

* to upgrade a single file, use:

  {cmd} <inputfile> [outputfile]

* to upgrade all .yml files in a directory, use:

  {cmd} <inputdirectory>

* to upgrade several files matching a pattern, use:

  {cmd} <inputpattern>

  In a pattern, the following characters are special:
      ?      matches any single character
      *      matches any number of characters
      [abc]  matches any character listed between the []
      [!abc] matches any character not listed between the []

  For example:

  {cmd} models/*.yml
  {cmd} */*.yml
  {cmd} examples/demo0?.yml
""".format(cmd=args[0]))
        sys.exit()
    try:
        upgrade(args[1], args[2] if len(args) > 2 else None)
    except ValueError as e:
        print(e)
