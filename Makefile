TESTS_DIR=liam2/tests

all: flake8 test

check-syntax-errors:
	@# This is a hack around flake8 not displaying E910 errors with the select option.
	@# Do not analyse .gitignored files.
	@# `make` needs `$$` to output `$`. Ref: http://stackoverflow.com/questions/2382764.
	test -z "`flake8 --first $(shell git ls-files | grep "\.py$$") | grep E901`"

clean: clean-build clean-cython clean-pyc

clean-build:
	rm -rf build

clean-cython:
	rm liam2/cpartition.c liam2/cutils.c cpartition.so cutils.so

clean-pyc:
	find . -name '*.pyc' -exec rm \{\} \;

ctags:
	ctags --recurse=yes .

flake8: clean-pyc
	flake8

test: check-syntax-errors
	nosetests $(TESTS_DIR) --exe --stop --with-doctest

test-ci: check-syntax-errors
	nosetests $(TESTS_DIR) --exe --with-doctest

test-with-coverage:
	nosetests $(TESTS_DIR) --exe --stop --with-coverage --cover-package=openfisca_france --cover-erase --cover-branches --cover-html
