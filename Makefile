.DEFAULT_GOAL := help

SHELL=/bin/bash

UNAME=$(shell uname -s)
KERNEL=$(shell poetry version | cut -d' ' -f1)

.PHONY: install
install:  ## Install a virtual environment
	@poetry install -vv

.PHONY: kernel
kernel: install ## Create a kernel for jupyter lab
	@echo ${KERNEL}
	@poetry run pip install ipykernel
	@poetry run python -m ipykernel install --user --name=${KERNEL}


.PHONY: fmt
fmt:  ## Run autoformatting and linting
	@poetry run pre-commit run --all-files

.PHONY: test
test: install ## Run tests
	@poetry run pytest

.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@rm -rf .pytest_cache/
	@rm -rf .ruff_cache/
	@rm -f .coverage
	@rm -rf htmlcov
	@find . -type f -name '*.py[co]' -delete -or -type d -name __pycache__ -delete


.PHONY: coverage
coverage: ## test and coverage
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
