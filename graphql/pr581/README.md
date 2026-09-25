# Evaluation of biocypher PR #581

[PR #581](https://github.com/biocypher/biocypher/pull/581) adds a
`GraphQLSchemaGenerator` for [issue #500](https://github.com/biocypher/biocypher/issues/500)
(generate a Neo4j GraphQL Library schema from a BioCypher schema). This folder
tests it on the SBML part of the sys-bio-kgs graph, at PR commit
`852dae1451b81a38f4c8e62106adcc728a275096`, with `@neo4j/graphql` 7.6.3 and
Neo4j 5.26.

| File | What |
|---|---|
| `generate_schema.py` | Runs the PR's generator on the SBML entries of `config/simple_schema_config.yaml` |
| `pr581_sbml_no_source_target.graphql` | PR output before `source`/`target` were added to the SBML edges |
| `pr581_sbml.graphql` | PR output, unchanged |
| `pr581_sbml_minimal_fixes.graphql` | PR output with the three fixes needed to use it (see header) |
| `pr581_sbml_minimal_fixes.checks.js`, `.examples.graphql` | Checks against Cypher and example queries for it |
| `introspected_sbml.graphql` | Baseline: Neo4j's introspector run on the same database |

The hand-written schema it is compared with is the API's `../schema/sbml.graphql`.

## Reproduce

From the repository root, with the SBML graph deployed
(`PIPELINE_SCRIPT=create_knowledge_graph_sbml.py docker compose up -d`):

```bash
# generate with the PR's code, in a throwaway environment (the project stays on BioCypher 0.17)
uv run --no-project --with pyyaml \
    --with "biocypher @ git+https://github.com/iqb430/biocypher@852dae1451b81a38f4c8e62106adcc728a275096" \
    python graphql/pr581/generate_schema.py config/simple_schema_config.yaml graphql/pr581/pr581_sbml.graphql

cd graphql
node tools/check_schema.js pr581/pr581_sbml.graphql            # fails: see finding 2
node tools/test_queries.js pr581/pr581_sbml_minimal_fixes.graphql pr581/pr581_sbml_minimal_fixes.checks.js
node tools/test_queries.js schema/sbml.graphql schema/sbml.checks.js
node tools/introspect.js pr581/introspected_sbml.graphql
```

To try the PR schema in the Apollo Sandbox:
`GRAPHQL_SCHEMA=pr581/pr581_sbml_minimal_fixes.graphql docker compose --profile graphql up -d --no-deps --force-recreate graphql`

## Findings

| # | Finding | Evidence |
|---|---|---|
| 1 | Edges without `source`/`target` in the BioCypher schema are dropped without a warning; only an unused `...Props` type is generated. `source`/`target` had to be added to `reactant`, `product`, `modifier`, `contained entity` (now on `main`'s schema) for testing (documented limitation) | `pr581_sbml_no_source_target.graphql` |
| 2 | Schema is rejected by current `@neo4j/graphql`: `Directive "@relationshipProperties" may not be used on INTERFACE` (v3 syntax; v4+ uses `type`) | `check_schema.js pr581_sbml.graphql` |
| 3 | Relationship types are SCREAMING_SNAKE_CASE (`CONTAINED_ENTITY`), BioCypher writes PascalCase (`ContainedEntity`): every relationship query returns 0 (Cypher: 6 each) | fix 2 in `pr581_sbml_minimal_fixes.graphql` |
| 4 | Only outgoing relationship fields: a `Process` cannot list its reactants (`Cannot query field "reactant" on type "Process"`) | |
| 5 | `is_a` / `inherit_properties` ignored; no representation of the ontology hierarchy (the main motivation over the introspector in #500) | |
| 6 | Unknown types, including BioCypher arrays (`str[]`, `string[]`), become `String`: reading an annotation fails with `String cannot represent value: ["http://identifiers.org/uniprot:P03023"]` (the introspector generates `[String!]!`) | fix 3 in `pr581_sbml_minimal_fixes.graphql` |
| 7 | Field names are unreadable (`containedentity`, `isentityof`) | |
| 8 | Mutations are generated (12 create/update/delete fields); a knowledge graph API should default to read-only | |
| 9 | Not integrated: takes a raw dict, no BioCypher API, tests or docs; branch is 84 commits behind biocypher `main` | |

With fixes for 2, 3 and 6, all queries match Cypher and relationship
properties are available through the connection API.

Found along the way, not specific to the PR: nodes carry parent-class labels, so a
query on a parent class also returns subclass nodes (7 species instead of 6, also
with the introspector). A generator has to decide how to handle this, e.g. with
interfaces for parent classes.

| | PR #581 | Introspector | `../schema/sbml.graphql` |
|---|---|---|---|
| Builds in `@neo4j/graphql` 7 | no | yes | yes |
| Array properties | `String` (query error) | `[String!]!` | `[String!]` |
| Relationship queries | 0 results (yes after fixes) | yes | yes |
| Both directions | no | yes | yes |
| Type names = BioCypher classes | yes | no (`MaterialEntity` for the compartment) | yes |
| Properties without data (`sbo`, `stoichiometry`) | yes | no | yes |
| Read-only | no | no | yes |
| Hierarchy | no | no | root interface |
