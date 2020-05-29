# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import traceback
from itertools import chain

# we want debug output. This must be done before importing liam2
os.environ["DEBUG"] = "TRUE"

from liam2.compat import StringIO
from liam2.simulation import Simulation
from liam2.importer import csv2h5

use_travis = os.environ.get('USE_TRAVIS', None) == 'true'
test_root = os.path.abspath(os.path.dirname(__file__))


def run_simulation(fpath):
    # We should NOT override output_dir here because it breaks tests and examples which load back what they
    # write (e.g. demo_load.yml)
    simulation = Simulation.from_yaml(fpath, log_level='processes')
    simulation.run()


# we can't name this function anything containing "test" otherwise nosetests tries to execute it
def iterate_directory(directory, import_files, excluded_files):
    directory_path = os.path.join(test_root, directory)
    excluded_files = excluded_files + import_files

    # run import files before other tests to allow other tests to use created files
    for import_file in import_files:
        yield csv2h5, os.path.join(directory_path, import_file)

    for test_file in os.listdir(directory_path):
        if test_file.endswith('.yml') and test_file not in excluded_files:
            yield run_simulation, os.path.join(directory_path, test_file)


# test generator for nosetests (must return test_func, args)
def test_examples():
    import_files = ('demo_import.yml',)
    # Cannot display charts/pop up windows on Travis
    tests_needing_qt = ('demo02.yml', 'demo03.yml', 'demo04.yml', 'demo06.yml')
    excluded = tests_needing_qt if use_travis else ()
    for func, test_file in iterate_directory('examples', import_files, excluded):
        yield func, test_file


def test_functional():
    import_files = ('import_in_import.yml', 'import_issue154.yml', 'import.yml',)
    # those are not runnable by themselves
    excluded = ('imported1.yml', 'imported2.yml')
    # XXX: why are static.yml and generate.yml excluded???
    if use_travis:
        # exclude test_erf because scipy is a big dependency for a single function
        # exclude generate.yml because it does not test anything in itself but takes a few seconds
        excluded += ('test_erf.yml', 'generate.yml')
    for func, test_file in iterate_directory('functional', import_files, excluded):
        yield func, test_file


# =======================================================
# below is a basic test runner not depending on nosetests
# =======================================================

def print_title(s):
    print(s)
    print("=" * len(s))


def printnow(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def run_func(func, *args, **kwargs):
    verb = 'Importing' if func is csv2h5 else 'Running'
    printnow('{} {}...'.format(verb, os.path.relpath(args[0], test_root)), end=' ')

    # capture stdout and stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    try:
        func(*args, **kwargs)
        exc_type, exc_value, exc_traceback = None, None, None
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
    sys.stdout.seek(0)
    sys.stderr.seek(0)
    stdout_content = sys.stdout.read()
    stderr_content = sys.stderr.read()
    # restore original
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    if exc_value is None:
        printnow("done.")
        if stderr_content:
            printnow("STDERR\n======\n{}\n".format(stderr_content))
        return "ok"
    else:
        printnow("FAILED.\n")
        if stdout_content:
            printnow("STDOUT\n======\n{}\n".format(stdout_content))
        if stderr_content:
            printnow("STDERR\n======\n{}\n".format(stderr_content))
        # print errors to stdout instead of stderr to avoid PyCharm
        # randomly mixing up stdout and stderr output
        printnow("TRACEBACK\n=========")
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  file=sys.stdout)
        sys.stdout.flush()
        printnow()
        return "failed"


def run_tests():
    print_title('Using test root: {}'.format(test_root))
    all_tests = chain(test_examples(), test_functional())
    results = [run_func(func, test_file) for func, test_file in all_tests]
    num_failed = sum(r == "failed" for r in results)
    print()
    print("ran %d tests, %d failed" % (len(results), num_failed))
    if num_failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    run_tests()
