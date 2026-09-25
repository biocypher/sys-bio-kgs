# GraphQL API

A GraphQL API on the deployed Neo4j database, using the
[Neo4j GraphQL Library](https://neo4j.com/docs/graphql/current/) and Apollo Server.
For now it covers the SBML part of the graph.

| Path | Purpose |
|---|---|
| `index.js`, `package.json` | API server |
| `schema/sbml.graphql` | Schema for the SBML graph (hand-written, read-only), served by default |
| `schema/sbml.examples.graphql` | Example queries for it |
| `schema/sbml.checks.js` | Checks comparing its query results with Cypher |
| `tools/` | Helpers to validate schemas and run queries (see below) |
| `pr581/` | Evaluation of [biocypher PR #581](https://github.com/biocypher/biocypher/pull/581) (GraphQL schema generator), not part of the API |

## Run

With the database deployed (see "Docker Usage" in the main README):

```bash
docker compose --profile graphql up -d --no-deps graphql
```

Open http://localhost:4000 for the Apollo Sandbox, and paste queries from
`schema/sbml.examples.graphql`. `GRAPHQL_SCHEMA=<path relative to graphql/>`
serves a different schema; add `--force-recreate` to switch a running service.

The SBML-only graph is built with `create_knowledge_graph_sbml.py`:

```bash
PIPELINE_SCRIPT=create_knowledge_graph_sbml.py docker compose up -d
```

## Tools

Run from `graphql/` after `npm install` (Node >= 20). They connect to
`bolt://localhost:7687` unless `NEO4J_URI` is set.

| Command | Does |
|---|---|
| `node tools/check_schema.js <schema.graphql>` | Validate a schema with the Neo4j GraphQL Library (no database needed) |
| `node tools/test_queries.js <schema.graphql> <schema.checks.js>` | Compare GraphQL query results with Cypher counts |
| `node tools/run_examples.js <schema.graphql> <schema.examples.graphql>` | Run every example query and print the results |
| `node tools/introspect.js <out.graphql>` | Generate a schema from the database with Neo4j's introspector |

Each schema keeps its examples and checks next to it, as
`<name>.examples.graphql` and `<name>.checks.js`.

## Notes

- Nodes carry the labels of their parent classes, and the Neo4j GraphQL Library
  matches nodes by labels they have. A query on a parent class therefore also
  returns nodes of its subclasses: `physicalEntityRepresentations` includes the
  compartment, and the `SystemsBiologyRepresentation` interface returns it twice.
