#!/usr/bin/env bash

# Print commands being run
set -o xtrace

# Run checks
flake8 t3

pylint t3

pydocstyle t3

mypy t3