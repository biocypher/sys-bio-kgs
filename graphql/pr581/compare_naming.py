"""
Compare the names PR #581 generates with the labels and relationship types
BioCypher writes to Neo4j.

    uv run --no-project --with "biocypher==0.17.0" python graphql/pr581/compare_naming.py
"""

from biocypher._translate import Translator
from biocypher.output.write._batch_writer import parse_label


# copied from PR #581 (biocypher/_graphql.py, GraphQLSchemaGenerator)
def pr_pascal_case(name: str) -> str:
    name = name.replace("-", " ").replace("_", " ")
    return "".join(word.capitalize() for word in name.split())


def pr_screaming_snake_case(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_").upper()


def biocypher_label(name: str) -> str:
    """Label / relationship type as written by BioCypher's batch writer."""
    return Translator.name_sentence_to_pascal(parse_label(name))


NAMES = [
    "physical entity representation",
    "contained entity",
    "mRNA",
    "DNA sequence variant",
    "non-coding RNA",
    "SBML model",
    "protein_isoform",
]

if __name__ == "__main__":
    print(f"| Schema name | BioCypher writes | PR node type / label | PR relationship type |")
    print("|---|---|---|---|")
    for name in NAMES:
        bc = biocypher_label(name)
        node = pr_pascal_case(name)
        rel = pr_screaming_snake_case(name)
        mark = lambda x: f"`{x}`" + ("" if x == bc else " ❌")
        print(f"| {name} | `{bc}` | {mark(node)} | {mark(rel)} |")
