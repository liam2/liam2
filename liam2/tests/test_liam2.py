# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import traceback
from StringIO import StringIO


from itertools import chain

from liam2.simulation import Simulation
from liam2.importer import csv2h5

use_travis = os.environ.get('USE_TRAVIS', None) == 'true'

test_root = os.path.dirname(__file__)


def print_title(s):
    print(s)
    print("=" * len(s))


def simulate(test_file):
    output_dir = os.path.join(test_root, 'output')
    simulation = Simulation.from_yaml(test_file, output_dir=output_dir, log_level='processes')
    simulation.run()


def printnow(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def run_file(test_file):
    importing = 'import' in test_file
    verb = 'Importing' if importing else 'Running'
    func = csv2h5 if importing else simulate

    printnow('{} {}...'.format(verb, os.path.relpath(test_file, test_root)), end=' ')
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    exception = None
    try:
        func(test_file)
    except Exception, e:
        exception = e
    sys.stdout.seek(0)
    sys.stderr.seek(0)
    stdout_content = sys.stdout.read()
    stderr_content = sys.stderr.read()
    # restore original
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    if exception is None:
        printnow("done.")
        return "ok"
    else:
        printnow("FAILED.\n")
        if stdout_content:
            printnow("STDOUT\n======\n{}\n".format(stdout_content))
        if stderr_content:
            printnow("STDERR\n======\n{}\n".format(stderr_content))
        # print to stdout to avoid PyCharm randomly mixing up stdout and stderr output
        printnow("TRACEBACK\n=========")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        printnow()
        return "failed"


def iterate_directory(directory, dataset_creator, excluded_files):
    directory_path = os.path.join(test_root, directory)
    excluded_files = excluded_files + (dataset_creator,)
    yield os.path.join(directory_path, dataset_creator)
    for test_file in os.listdir(directory_path):
        if test_file.endswith('.yml') and test_file not in excluded_files:
            yield os.path.join(directory_path, test_file)


def test_examples():
    # No pyqt4 on travis
    need_qt = ('demo02.yml', 'demo03.yml', 'demo04.yml', 'demo06.yml')
    excluded = need_qt if use_travis else ()
    for test_file in iterate_directory('examples', 'demo_import.yml', excluded):
        yield test_file


def test_functional():
    excluded = ('imported1.yml', 'imported2.yml')
    if use_travis:
        excluded += ('static.yml', 'generate.yml')
    for test_file in iterate_directory('functional', 'import.yml', excluded):
        yield test_file


if __name__ == '__main__':
    print_title('Using test root: {}'.format(test_root))
    results = [run_file(f) for f in chain(test_examples(), test_functional())]
    if any(r == "failed" for r in results):
        sys.exit(1)
