# Set the default target to help
.DEFAULT_GOAL := help

# Create a Python virtual environment using uv
venv:
	@curl -LsSf https://astral.sh/uv/install.sh | sh  # Install uv package manager
	@uv venv --python='3.12'  # Create a virtual environment with Python 3.12


# Declare install as a phony target (not a file)
.PHONY: install
install: venv ## Install a virtual environment
	@uv pip install --upgrade pip  # Ensure pip is up to date
	@uv sync --all-extras --dev --frozen  # Install dependencies from pyproject.toml


# Declare fmt as a phony target
.PHONY: fmt
fmt: venv ## Run autoformatting and linting
	@uv pip install pre-commit  # Install pre-commit hooks
	@uv run pre-commit install  # Set up pre-commit hooks
	@uv run pre-commit run --all-files  # Run pre-commit hooks on all files


# Declare clean as a phony target
.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@git clean -X -d -f  # Remove files ignored by git


# Declare help as a phony target
.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"  # Print header in bold
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort  # Extract and format targets with comments


# Declare marimo as a phony target
.PHONY: marimo
marimo: install ## Install Marimo
	@uv pip install marimo  # Install Marimo
	@uv run marimo edit book/marimo  # Start Marimo server


# Declare test as a phony target
.PHONY: test
test: install  ## Run pytests
	@uv pip install pytest  # Install pytest
	@uv run pytest tests  # Run tests in tests directory


# Declare jupyter as a phony target
.PHONY: jupyter
jupyter: install ## Run jupyter lab
	@uv pip install jupyterlab  # Install JupyterLab
	@uv run jupyter lab  # Start JupyterLab server
