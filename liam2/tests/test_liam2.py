# -*- coding: utf-8 -*-


from __future__ import print_function

import os
import pkg_resources

import liam2
from liam2.simulation import Simulation


def run_simulation(yaml_file, input_dir, output_dir):
    try:
        simulation = Simulation.from_yaml(
            yaml_file,
            input_dir = input_dir,
            output_dir = output_dir,
            )
        simulation.run(False)
    except Exception as e:
        print(e)
        assert False, '{} failed'.format(yaml_file)


def test_liam2_examples_files():
    liam2_examples_directory = os.path.join(
        pkg_resources.get_distribution('liam2').location,
        'liam2',
        'tests',
        'examples'
        )
    excluded_files = [
        'demo_import.yml',  # non working example
        'demo02.yml',  # TODO: pb with figures
        ]
    yaml_files = [os.path.join(liam2_examples_directory, _file) for _file in os.listdir(liam2_examples_directory)
        if os.path.isfile(os.path.join(liam2_examples_directory, _file))
        and _file.endswith('.yml')
        and _file not in excluded_files]

    input_dir = os.path.join(liam2_examples_directory)
    output_dir = os.path.join(
        pkg_resources.get_distribution('liam2').location,
        'liam2',
        'tests',
        'output',
        )
    for yaml_file in yaml_files:
        yield run_simulation, yaml_file, input_dir, output_dir


def test_liam2_functionnal_files():
    liam2_functional_directory = os.path.join(
        pkg_resources.get_distribution('liam2').location,
        'liam2',
        'tests',
        'functional'
        )
    # TODO does not work
    import_yaml_path = os.path.join(liam2_functional_directory, 'import.yml')
    assert os.path.exists(import_yaml_path), '{} does not exists'.format(import_yaml_path)
    simulation_yaml_path = os.path.join(liam2_functional_directory, 'simulation.yml')
    assert os.path.exists(simulation_yaml_path), '{} does not exists'.format(simulation_yaml_path)

    input_dir = os.path.join(liam2_functional_directory)
    output_dir = os.path.join(
        pkg_resources.get_distribution('liam2').location,
        'tests',
        'output'
        )
    yaml_files = [import_yaml_path, simulation_yaml_path]
    for yaml_file in yaml_files:
        yield run_simulation, yaml_file, input_dir, output_dir
