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


def run_file(test_file):
    if 'import' in test_file:
        csv2h5(test_file)
    else:
        # We should NOT override output_dir here because it breaks tests and examples which load back what they
        # write (e.g. demo_load.yml)
        simulation = Simulation.from_yaml(test_file, log_level='processes')
        simulation.run()


def iterate_directory(directory, dataset_creator, excluded_files):
    directory_path = os.path.join(test_root, directory)
    excluded_files = excluded_files + (dataset_creator,)
    yield os.path.join(directory_path, dataset_creator)
    for test_file in os.listdir(directory_path):
        if test_file.endswith('.yml') and test_file not in excluded_files:
            yield os.path.join(directory_path, test_file)


# test generator for nosetests (must return test_func, args)
def test_examples():
    # Cannot display charts/pop up windows on Travis
    need_qt = ('demo02.yml', 'demo03.yml', 'demo04.yml', 'demo06.yml')
    excluded = need_qt if use_travis else ()
    for test_file in iterate_directory('examples', 'demo_import.yml', excluded):
        yield run_file, test_file


def test_functional():
    excluded = ('imported1.yml', 'imported2.yml')
    if use_travis:
        excluded += ('test_erf.yml', 'static.yml', 'generate.yml')
    for test_file in iterate_directory('functional', 'import.yml', excluded):
        yield run_file, test_file


if __name__ == '__main__':
    def print_title(s):
        print(s)
        print("=" * len(s))


    def printnow(*args, **kwargs):
        print(*args, **kwargs)
        sys.stdout.flush()


    def run_func(func, *args, **kwargs):
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
            # print to stdout to avoid PyCharm randomly mixing up stdout and stderr output
            printnow("TRACEBACK\n=========")
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      file=sys.stdout)
            sys.stdout.flush()
            printnow()
            return "failed"


    print_title('Using test root: {}'.format(test_root))

    results = []
    for test_func, test_file in chain(test_examples(), test_functional()):
        verb = 'Importing' if 'import' in test_file else 'Running'
        printnow('{} {}...'.format(verb, os.path.relpath(test_file, test_root)), end=' ')
        results.append(run_func(test_func, test_file))
    num_failed = sum(r == "failed" for r in results)
    print()
    print("ran %d tests, %d failed" % (len(results), num_failed))
    if num_failed > 0:
        sys.exit(1)
