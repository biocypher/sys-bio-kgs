// Validate GraphQL type definitions with the Neo4j GraphQL Library (no database needed).
// Usage (from graphql/): node tools/check_schema.js <schema.graphql>
import { readFileSync } from "node:fs";

import { Neo4jGraphQL } from "@neo4j/graphql";

const path = process.argv[2];
try {
  await new Neo4jGraphQL({ typeDefs: readFileSync(path, "utf-8") }).getSchema();
  console.log(`OK: ${path}`);
} catch (error) {
  const errors = Array.isArray(error) ? error : [error];
  const messages = [...new Set(errors.map((e) => e.message))];
  console.log(`INVALID: ${path} (${errors.length} errors)`);
  for (const message of messages) console.log(`  - ${message}`);
  process.exitCode = 1;
}
