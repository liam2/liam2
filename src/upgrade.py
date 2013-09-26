from __future__ import print_function

import os.path
import re

__version__ = '0.2'


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

    # transform *link(linkname, ...) -> linkname.*(...)
    content = re.sub("([a-zA-Z]+)link\s*\(\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*,?\s*",
                     r"\2.\1(",
                     content)

    # grpmin(expr1, expr2, ...) -> min(expr1, filter=expr2, ...)
    # grpmax(expr1, expr2, ...) -> max(expr1, filter=expr2, ...)
    def repl(obj):
        # the 4th group is optional, so we change it from None to ''
        groups = [s or '' for s in obj.groups()]
        return '{}({}, filter={}{})'.format(*groups)
    pattern = "grp(max|min)\s*\(([^),]+),\s*([^,)=]+)(,\s*[^)]+\s*)?\)"
    content = re.sub(pattern, repl, content)

    # grpXXX(...) -> XXX(...)
    content = re.sub("grp([a-zA-Z]+)\s*\(",
                     r"\1(",
                     content)

    with open(outpath, "wb") as f:
        f.write(content)
    print("upgraded model written to: '%s'" % outpath)

if __name__ == '__main__':
    import sys

    print("LIAM2 model upgrader %s\n" % __version__)

    args = sys.argv
    if len(args) < 2:
        print("Usage: %s inputfile [outputfile]" % args[0])
        sys.exit()

    upgrade(args[1], args[2] if len(args) > 2 else None)
