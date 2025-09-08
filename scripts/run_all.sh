#!/bin/bash
python -m mlproject.cli prepare "$@"
python -m mlproject.cli train "$@"
python -m mlproject.cli evaluate "$@"
