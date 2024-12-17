.DEFAULT_GOAL := help

SHELL=/bin/bash

UNAME=$(shell uname -s)

.PHONY: install
install:  ## Install a virtual environment
	@poetry install -vv

.PHONY: fmt
fmt:  ## Run autoformatting and linting
	@poetry run pip install pre-commit
	@poetry run pre-commit install
	@poetry run pre-commit run --all-files

.PHONY: test
test: install ## Run tests
	@poetry run pytest

.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@git clean -X -d -f


.PHONY: coverage
coverage: install ## test and coverage
	@poetry run coverage run --source=cvx/. -m pytest
	@poetry run coverage report -m
	@poetry run coverage html

	@if [ ${UNAME} == "Darwin" ]; then \
		open htmlcov/index.html; \
	elif [ ${UNAME} == "linux" ]; then \
		xdg-open htmlcov/index.html 2> /dev/null; \
	fi


.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort


.PHONY: jupyter
jupyter: install ## Run jupyter lab
	@poetry run pip install jupyterlab
	@poetry run jupyter lab


.PHONY: boil
boil: ## Update the boilerplate code
	@poetry run pip install cvxcooker
	@poetry run cook pyproject.toml


.PHONY: marimo
marimo: install ## Run jupyter lab
	@poetry run pip install marimo
	@poetry run marimo edit book/marimo
