// Generate GraphQL type definitions from the live database with Neo4j's introspector,
// e.g. as a baseline to compare other schemas with.
// Usage (from graphql/): node tools/introspect.js <out.graphql>
//   NEO4J_URI  bolt URI of the database (default: bolt://localhost:7687)
import { writeFileSync } from "node:fs";

import { toGraphQLTypeDefs } from "@neo4j/introspector";
import neo4j from "neo4j-driver";

const driver = neo4j.driver(process.env.NEO4J_URI ?? "bolt://localhost:7687");
const sessionFactory = () => driver.session({ defaultAccessMode: neo4j.session.READ });

const typeDefs = await toGraphQLTypeDefs(sessionFactory);
writeFileSync(process.argv[2], typeDefs);
console.log(`Wrote ${process.argv[2]}`);
await driver.close();
