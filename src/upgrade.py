from __future__ import print_function

import re

__version__ = '0.1'


def convert(inpath, outpath=None):
    if outpath is None:
        outpath = inpath + ".upgraded"
    with open(inpath, "rb") as f:
        content = f.read()
    print("model read from: '%s'" % inpath)
    content = re.sub("([a-zA-Z]+)link\s*\(\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*,?\s*",
                     r"\2.\1(",
                     content)
    with open(outpath, "wb") as f:
        f.write(content)
    print("model written to: '%s'" % outpath)

if __name__ == '__main__':
    import sys

    print("LIAM2 model upgrader %s\n" % __version__)

    args = sys.argv
    if len(args) < 2:
        print("Usage: %s inputfile [outputfile]" % args[0])
        sys.exit()

    convert(args[1], args[2] if len(args) > 2 else None)
