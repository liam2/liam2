from __future__ import print_function

import os.path
import re

__version__ = '0.2'


def upgrade_str(content):
    # transform *link(linkname, ...) -> linkname.*(...)
    content = re.sub("([a-zA-Z]+)link\s*\(\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*,?\s*",
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

    # grpXYZ(...) -> XYZ(...)
    return re.sub("grp([a-zA-Z]+)\s*\(",
                  r"\1(",
                  content)


def upgrade(inpath, outpath=None):
    if outpath is None:
        outpath = inpath

    print("original model read from: '%s'" % inpath)
    with open(inpath, "rb") as f:
        content = f.read()

    if outpath == inpath:
        filename, _ = os.path.splitext(inpath)
        backup_path = filename + ".bak"
        print("original model copied to: '%s'" % backup_path)
        with open(backup_path, "wb") as f:
            f.write(content)

    with open(outpath, "wb") as f:
        f.write(upgrade_str(content))
    print("upgraded model written to: '%s'" % outpath)

if __name__ == '__main__':
    import sys

    print("LIAM2 model upgrader %s\n" % __version__)

    args = sys.argv
    if len(args) < 2:
        print("Usage: %s inputfile [outputfile]" % args[0])
        sys.exit()

    upgrade(args[1], args[2] if len(args) > 2 else None)
