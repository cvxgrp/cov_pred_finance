#!/bin/bash

name="covariance"

poetry install
poetry run pip install ipykernel
poetry run python -m ipykernel install --user --name=$name
