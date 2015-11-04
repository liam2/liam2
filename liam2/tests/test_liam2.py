# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import pkg_resources
from itertools import chain

from liam2.simulation import Simulation
from liam2.importer import csv2h5

use_travis = os.environ.get('USE_TRAVIS', None) == 'true'

test_root = os.path.join(
    pkg_resources.get_distribution('liam2').location,
    'liam2',
    'tests'
    )


def run_simulation(yaml_file, output_dir):
    simulation = Simulation.from_yaml(yaml_file, output_dir=output_dir)
    print('Running {} using {} as output dir'.format(yaml_file, output_dir))
    simulation.run()


def test_functional():
    functional_excluded_files = ['imported1.yml', 'imported2.yml']
    if use_travis:
        functional_excluded_files.extend(['static.yml', 'generate.yml'])
    for test_file in iterate_files('functional', functional_excluded_files):
        yield run_file, test_file


def test_examples():
    # No pyqt4 on travis
    need_qt = ['demo02.yml', 'demo03.yml', 'demo04.yml', 'demo06.yml']
    examples_excluded_files = need_qt if use_travis else []
    for test_file in iterate_files('examples', examples_excluded_files):
        yield run_file, test_file


def iterate_files(folder_name, excluded_files = None):
    test_folder = os.path.join(test_root, folder_name)
    test_fnames = [f for f in os.listdir(test_folder)
                   if f.endswith('.yml') and f not in excluded_files]
    test_files = [os.path.join(test_folder, f) for f in test_fnames]

    for test_file in test_files:
        yield test_file


def run_file(test_file):
    output_dir = os.path.join(test_root, 'output')
    if 'import' in test_file:
        print("import", test_file)
        csv2h5(test_file)
    else:
        print("run", test_file)
        run_simulation(test_file, output_dir)


if __name__ == '__main__':
   for test in chain(test_examples(), test_functional()):
        func, args = test[0], test[1:]
        func(*args)
