// Run GraphQL queries against the live database and compare with Cypher.
// Usage (from graphql/): node tools/test_queries.js <schema.graphql> <schema.checks.js>
//   checks.js exports `checks`: [description, GraphQL query, count(data), Cypher count query]
//   NEO4J_URI  bolt URI of the database (default: bolt://localhost:7687)
import { readFileSync } from "node:fs";

import { Neo4jGraphQL } from "@neo4j/graphql";
import { graphql } from "graphql";
import neo4j from "neo4j-driver";

const driver = neo4j.driver(process.env.NEO4J_URI ?? "bolt://localhost:7687");
const typeDefs = readFileSync(process.argv[2], "utf-8");
const schema = await new Neo4jGraphQL({ typeDefs, driver }).getSchema();

async function cypherCount(query) {
  const { records } = await driver.executeQuery(query);
  return records[0].get(0).toNumber();
}

async function gql(source) {
  const result = await graphql({ schema, source, contextValue: {} });
  return result;
}

const { checks } = await import(new URL(process.argv[3], `file://${process.cwd()}/`).href);

for (const [description, query, count, cypher] of checks) {
  const expected = await cypherCount(cypher);
  const result = await gql(query);
  let status;
  if (result.errors) {
    status = `ERROR: ${[...new Set(result.errors.map((e) => e.message))].join("; ")}`;
  } else {
    const got = count(result.data);
    status = got === expected ? `ok (${got})` : `MISMATCH: GraphQL ${got}, Cypher ${expected}`;
  }
  console.log(`${description.padEnd(42)} ${status}`);
}
await driver.close();
