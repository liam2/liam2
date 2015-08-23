# -*- coding: utf-8 -*-


from __future__ import print_function

import os
import pkg_resources

import liam2
from liam2.simulation import Simulation
from liam2.importer import file2h5

def run_simulation(yaml_file, input_dir, output_dir):
    # try:
        simulation = Simulation.from_yaml(
            yaml_file,
            input_dir = input_dir,
            output_dir = output_dir,
            )
        print('About to run simulation file {} using {} as input dir and {} as output_dir'.format(
            yaml_file, input_dir, output_dir
            ))
        simulation.run(run_console=False)
    # except Exception:
    #     print('{} failed'.format(yaml_file))
    #     raise


def test_liam2(subfolder, excluded_files):
    liam2_test_directory = os.path.join(
        pkg_resources.get_distribution('liam2').location,
        'liam2',
        'tests',
        subfolder
        )
    yaml_files = [os.path.join(liam2_test_directory, _file) for _file in os.listdir(liam2_test_directory)
        if os.path.isfile(os.path.join(liam2_test_directory, _file))
        and _file.endswith('.yml')
        and _file not in excluded_files]

    input_dir = os.path.join(liam2_test_directory)
    output_dir = os.path.join(
        pkg_resources.get_distribution('liam2').location,
        'liam2',
        'tests',
        'output',
        )
    for yaml_file in yaml_files:
        if 'import' in yaml_file:
            yield file2h5, yaml_file, input_dir, (10 * 2 ** 20)
        else:
            yield run_simulation, yaml_file, input_dir, output_dir    

def test_liam2_examples_files():
    return test_liam2('examples', [])
    
    
def test_liam2_functionnal_files():
    return test_liam2('functional', ['imported1.yml', 'imported2.yml'])



if __name__ == '__main__':
    for function, yaml_file, input_dir, output_dir in test_liam2_examples_files():
        print (yaml_file)
        function(yaml_file, input_dir, output_dir)

    for function, yaml_file, input_dir, output_dir in test_liam2_functionnal_files():
        print(yaml_file)
        function(yaml_file, input_dir, output_dir)