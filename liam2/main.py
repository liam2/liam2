# encoding: utf-8
from __future__ import print_function

import argparse
import os
from os.path import splitext
import platform
import traceback
import sys
import warnings

# this is needed for vitables and needs to happen BEFORE matplotlib is
# imported (and imports PyQt)
import sip
sip.setapi('QString', 2)
sip.setapi('QVariant', 2)

import yaml

import config
from importer import file2h5
from console import Console
from context import EvaluationContext
from data import entities_from_h5, H5Data
from simulation import Simulation
from upgrade import upgrade
from utils import AutoFlushFile
from view import viewhdf

__version__ = "0.10.2"


def write_traceback(e):
    try:
        import traceback
        # output_directory might not be set at this point yet and it is
        # only set for run and explore commands but when it is not set
        # its default value of "." is used and thus we get the "old"
        # behaviour: error.log in the working directory
        out_dir = config.output_directory
        error_path = os.path.join(out_dir, 'error.log')
        error_path = os.path.abspath(error_path)
        with file(error_path, 'w') as f:
            traceback.print_exc(file=f)
            if hasattr(e, 'liam2context'):
                f.write(e.liam2context)
        return error_path
    except IOError, log_ex:
        print("WARNING: could not save technical error log "
              "({} on '{}')".format(log_ex.strerror, log_ex.filename))
    except Exception, log_ex:
        print(log_ex)
    return None


def printerr(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def print_exception_wh_context(ex_type, e, tb):
    traceback.print_exception(ex_type, e, tb, file=sys.stderr)
    if hasattr(e, 'liam2context'):
        printerr(e.liam2context)


def print_exception_simplified(ex_type, e, tb):
    # e.context      | while parsing a block mapping
    # e.context_mark | in "import.yml", line 18, column 9
    # e.problem      | expected <block end>, but found '<block sequence start>'
    # e.problem_mark | in "import.yml", line 29, column 12
    error_log_path = write_traceback(e)
    if isinstance(e, yaml.parser.ParserError):
        # eg, inconsistent spacing, no space after a - in a list, ...
        printerr("SYNTAX ERROR {}".format(str(e.problem_mark).strip()))
    elif isinstance(e, yaml.scanner.ScannerError):
        # eg, tabs, missing colon for mapping. The reported problem is
        # different when it happens on the first line (no context_mark) and
        # when it happens on a subsequent line.
        if e.context_mark is not None:
            msg = e.problem if e.problem != "could not found expected ':'" \
                else "could not find expected ':'"
            mark = e.context_mark
        else:
            if (e.problem ==
                    "found character '\\t' that cannot start any token"):
                msg = "found a TAB character instead of spaces"
            else:
                msg = ""
            mark = e.problem_mark
        if msg:
            msg = ": " + msg
        printerr("SYNTAX ERROR {}{}".format(str(mark).strip(), msg))
    elif isinstance(e, yaml.reader.ReaderError):
        if e.encoding == 'utf8':
            printerr("\nERROR in '{}': invalid character found, this probably "
                     "means you have used non ASCII characters (accents and "
                     "other non-english characters) and did not save your file "
                     "using the UTF8 encoding".format(e.name))
        else:
            printerr("\nERROR:", str(e))
    elif isinstance(e, SyntaxError):
        printerr("SYNTAX ERROR:", e.msg.replace('EOF', 'end of block'))
        if e.text is not None:
            printerr(e.text)
            offset_str = ' ' * (e.offset - 1) if e.offset > 0 else ''
            printerr(offset_str + '^')
    else:
        printerr("\nERROR:", str(e))

    if hasattr(e, 'liam2context'):
        printerr(e.liam2context)

    if error_log_path is not None:
        printerr()
        printerr("the technical error log can be found at", error_log_path)


def simulate(args):
    print("Using simulation file: '{}'".format(args.fpath))

    simulation = Simulation.from_yaml(args.fpath,
                                      input_dir=args.input_path,
                                      input_file=args.input_file,
                                      output_dir=args.output_path,
                                      output_file=args.output_file,
                                      start_period=args.startperiod,
                                      periods=args.periods, seed=args.seed,
                                      skip_shows=args.skipshows,
                                      skip_timings=args.skiptimings,
                                      log_level=args.loglevel,
                                      assertions=args.assertions,
                                      autodump=args.autodump,
                                      autodiff=args.autodiff)

    simulation.run(args.interactive)
#    import cProfile as profile
#    profile.runctx('simulation.run(args.interactive)', vars(), {},
#                   'c:\\tmp\\simulation.profile')
    # to use profiling data:
    # import pstats
    # p = pstats.Stats('c:\\tmp\\simulation.profile')
    # p.strip_dirs().sort_stats('cum').print_stats(30)


def explore(fpath):
    _, ext = splitext(fpath)
    ftype = 'data' if ext in ('.h5', '.hdf5') else 'simulation'
    print("Using {} file: '{}'".format(ftype, fpath))
    if ftype == 'data':
        globals_def, entities = entities_from_h5(fpath)
        data_source = H5Data(None, fpath)
        h5in, _, globals_data = data_source.load(globals_def, entities)
        h5out = None
        simulation = Simulation(globals_def, None, None, None, None,
                                entities.values(), None)
        period, entity_name = None, None
    else:
        simulation = Simulation.from_yaml(fpath)
        h5in, h5out, globals_data = simulation.load()
        period = simulation.start_period + simulation.periods - 1
        entity_name = simulation.default_entity
    entities = simulation.entities_map
    if entity_name is None and len(entities) == 1:
        entity_name = entities.keys()[0]
    if period is None and entity_name is not None:
        entity = entities[entity_name]
        period = max(entity.output_index.keys())
    eval_ctx = EvaluationContext(simulation, entities, globals_data, period,
                                 entity_name)
    try:
        c = Console(eval_ctx)
        c.run()
    finally:
        h5in.close()
        if h5out is not None:
            h5out.close()


def display(fpath):
    print("Launching ViTables...")
    _, ext = splitext(fpath)
    if ext in ('.h5', '.hdf5'):
        files = [fpath]
    else:
        ds = Simulation.from_yaml(fpath).data_source
        files = [ds.input_path, ds.output_path]
    print("Trying to open:", " and ".join(str(f) for f in files))
    viewhdf(files)


class PrintVersionsAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        import numpy
        import numexpr
        import tables

        try:
            from cpartition import filter_to_indices

            del filter_to_indices
            cext = True
        except ImportError:
            cext = False
        print("C extensions are" + (" NOT" if not cext else "") + " available")

        # optional dependencies
        try:
            import vitables
            vt_version = vitables.__version__
        except ImportError:
            vt_version = 'N/A'

        try:
            import matplotlib
            mpl_version = matplotlib.__version__
        except ImportError:
            mpl_version = 'N/A'

        try:
            import bcolz
            bcolz_version = bcolz.__version__
        except ImportError:
            bcolz_version = 'N/A'

        py_version = '{} ({})'.format(platform.python_version(),
                                      platform.architecture()[0])
        print("""
python {py}
numpy {np}
numexpr {ne}
pytables {pt}
bcolz {bc}
pyyaml {yml}
vitables {vt}
matplotlib {mpl}
""".format(py=py_version, np=numpy.__version__, ne=numexpr.__version__,
           pt=tables.__version__, vt=vt_version, mpl=mpl_version,
           bc=bcolz_version, yml=yaml.__version__))
        parser.exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--versions', action=PrintVersionsAction, nargs=0,
                        help="display versions of dependencies")
    parser.add_argument('--debug', action='store_true', default=False,
                        help="run in debug mode")
    parser.add_argument('--input-path', dest='input_path',
                        help='override the input path')
    parser.add_argument('--input-file', dest='input_file',
                        help='override the input file')
    parser.add_argument('--output-path', dest='output_path',
                        help='override the output path')
    parser.add_argument('--output-file', dest='output_file',
                        help='override the output file')

    subparsers = parser.add_subparsers(dest='action')

    # create the parser for the "run" command
    parser_run = subparsers.add_parser('run', help='run a simulation')
    parser_run.add_argument('fpath', help='simulation file')
    parser_run.add_argument('-i', '--interactive', action='store_true',
                            help='show the interactive console after the '
                                 'simulation')
    parser_run.add_argument('-sp', '--startperiod', type=int,
                            help='first period to simulate (integer)')
    parser_run.add_argument('-p', '--periods', type=int,
                            help='number of periods to simulate (integer)')
    parser_run.add_argument('-s', '--seed', type=int,
                            help='defines the starting point of the '
                                 'pseudo-random generator (integer)')
    parser_run.add_argument('-ss', '--skipshows', action='store_true',
                            help='do not log shows')
    parser_run.add_argument('-st', '--skiptimings', action='store_true',
                            help='do not log timings')
    parser_run.add_argument('-ll', '--loglevel',
                            choices=['periods', 'functions', 'processes'],
                            help='defines the logging level')
    parser_run.add_argument('--autodump', help='path of the autodump file')
    parser_run.add_argument('--autodiff', help='path of the autodiff file')
    parser_run.add_argument('--assertions', choices=['raise', 'warn', 'skip'],
                            help='determines behavior of assertions')

    # create the parser for the "import" command
    parser_import = subparsers.add_parser('import', help='import data')
    parser_import.add_argument('file', help='import file')

    # create the parser for the "explore" command
    parser_explore = subparsers.add_parser('explore', help='explore data of a '
                                                           'past simulation')
    parser_explore.add_argument('file', help='explore file')

    # create the parser for the "upgrade" command
    parser_upgrade = subparsers.add_parser('upgrade',
                                           help='upgrade a simulation file to '
                                                'the latest syntax')
    parser_upgrade.add_argument('input', help='input simulation file')
    out_help = "output simulation file. If missing, the original file will " \
               "be backed up (to filename.bak) and the upgrade will be " \
               "done in-place."
    parser_upgrade.add_argument('output', help=out_help, nargs='?')

    # create the parser for the "view" command
    parser_import = subparsers.add_parser('view', help='view data')
    parser_import.add_argument('file', help='data file')

    parsed_args = parser.parse_args()
    if parsed_args.debug:
        config.debug = True

    # this can happen via the environment variable too!
    if config.debug:
        # by default, DeprecationWarning and PendingDeprecationWarning, and
        # ImportWarning are ignored, this shows them.
        warnings.simplefilter('default')
        sys.excepthook = print_exception_wh_context
    else:
        sys.excepthook = print_exception_simplified

    action = parsed_args.action
    if action == 'run':
        func, args = simulate, (parsed_args,)
    elif action == "import":
        args = file2h5, parsed_args.file
    elif action == "explore":
        func, args = explore, (parsed_args.file,)
    elif action == "upgrade":
        func, args = upgrade, (parsed_args.input, parsed_args.output)
    elif action == "view":
        func, args = display, (parsed_args.file,)
    else:
        raise ValueError("invalid action: {}".format(action))
    return func(*args)


if __name__ == '__main__':
    sys.stdout = AutoFlushFile(sys.stdout)
    sys.stderr = AutoFlushFile(sys.stderr)

    print("LIAM2 {} ({})".format(__version__, platform.architecture()[0]))
    print()

    main()
