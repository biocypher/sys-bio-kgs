"""
Generate a Neo4j GraphQL schema for the SBML part of the BioCypher schema,
using GraphQLSchemaGenerator from biocypher PR #581.

Run in a throwaway environment with the PR's biocypher (keeps the project on 0.17):

    uv run --no-project --with pyyaml \
        --with "biocypher @ git+https://github.com/iqb430/biocypher@852dae1451b81a38f4c8e62106adcc728a275096" \
        python graphql/pr581/generate_schema.py config/simple_schema_config.yaml graphql/pr581/pr581_sbml.graphql
"""

import sys

import yaml
from biocypher._graphql import GraphQLSchemaGenerator

# BioCypher schema entries used by the SBML adapter
SBML_ENTITIES = [
    "model",
    "process",
    "physical entity representation",
    "physical compartment",
    "reactant",
    "product",
    "modifier",
    "contained entity",
    "is entity of",
    "is process of",
    "is compartment of",
]


def main(schema_path, out_path):
    schema = yaml.safe_load(open(schema_path))
    sbml_schema = {name: schema[name] for name in SBML_ENTITIES}
    graphql = GraphQLSchemaGenerator(sbml_schema).generate()
    with open(out_path, "w") as f:
        f.write(graphql)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
