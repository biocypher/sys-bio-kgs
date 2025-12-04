#!/bin/bash -c
cd /usr/app/
cp -r /src/* .
cp config/biocypher_docker_config.yaml config/biocypher_config.yaml
export UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
pip install uv
uv sync --python 3.11 --extra momapy_sbml_kinetic --index-strategy unsafe-best-match
uv run python3 create_knowledge_graph_sbgn_sbml.py
chmod -R 777 biocypher-log