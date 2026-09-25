// Run every named operation in an examples file against the live database.
// Usage (from graphql/): node tools/run_examples.js <schema.graphql> <schema.examples.graphql>
//   NEO4J_URI  bolt URI of the database (default: bolt://localhost:7687)
import { readFileSync } from "node:fs";

import { Neo4jGraphQL } from "@neo4j/graphql";
import { graphql, parse } from "graphql";
import neo4j from "neo4j-driver";

const driver = neo4j.driver(process.env.NEO4J_URI ?? "bolt://localhost:7687");
const schema = await new Neo4jGraphQL({ typeDefs: readFileSync(process.argv[2], "utf-8"), driver }).getSchema();
const source = readFileSync(process.argv[3], "utf-8");

for (const op of parse(source).definitions) {
  const name = op.name.value;
  const result = await graphql({ schema, source, operationName: name, contextValue: {} });
  const output = result.errors
    ? `ERROR: ${result.errors.map((e) => e.message).join("; ")}`
    : JSON.stringify(result.data);
  console.log(`${name}: ${output.slice(0, 160)}${output.length > 160 ? "..." : ""}`);
}
await driver.close();
