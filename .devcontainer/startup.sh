#!/bin/bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync -vv --frozen
