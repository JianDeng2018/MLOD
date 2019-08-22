#!/usr/bin/env bash

cd "$(dirname "$0")"
cd ../..

echo "Running example unit tests"
coverage run --source mlod -m unittest discover -b --pattern "test_*.py"
#unittest discover -b --pattern "test_*.py"

echo "Running unit tests in $(pwd)"
coverage run --source mlod -m unittest discover -b --pattern "*_test.py"

#coverage report -m
