# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import pkg_resources

from liam2.simulation import Simulation
from liam2.importer import csv2h5

use_travis = os.environ.get('USE_TRAVIS', None) == 'true'


def run_simulation(yaml_file, output_dir):
    simulation = Simulation.from_yaml(yaml_file, output_dir=output_dir)
    print('Running {} using {} as output dir'.format(yaml_file, output_dir))
    simulation.run()


def test_liam2():
    need_qt = ['demo02.yml', 'demo03.yml', 'demo04.yml']
    # No pyqt4 on travis
    examples_excluded_files = need_qt if use_travis else []
    functional_excluded_files = ['imported1.yml', 'imported2.yml']
    if use_travis:
        functional_excluded_files.extend(['static.yml', 'generate.yml'])

    test_folders = {
        'examples': examples_excluded_files,
        'functional': functional_excluded_files
    }

    test_root = os.path.join(
        pkg_resources.get_distribution('liam2').location,
        'liam2',
        'tests'
    )
    for folder_name, excluded_files in test_folders.items():
        test_folder = os.path.join(test_root, folder_name)
        test_fnames = [f for f in os.listdir(test_folder)
                       if f.endswith('.yml') and f not in excluded_files]
        test_files = [os.path.join(test_folder, f) for f in test_fnames]

        output_dir = os.path.join(test_root, 'output')
        for test_file in test_files:
            if 'import' in test_file:
                print("import", test_file)
                yield csv2h5, test_file
            else:
                print("run", test_file)
                yield run_simulation, test_file, output_dir

if __name__ == '__main__':
    for test in test_liam2():
        func = test[0]
        args = test[1:]
        func(*args)
