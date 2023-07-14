#!/bin/bash
pipx install poetry
poetry config virtualenvs.in-project true
poetry install
poetry run pre-commit install
