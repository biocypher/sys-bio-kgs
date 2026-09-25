// GraphQL API on the Neo4j database, using the Neo4j GraphQL Library.
//
//   NEO4J_URI  bolt URI of the database (default: bolt://localhost:7687)
//   SCHEMA     GraphQL type definitions to serve (default: schema/sbml.graphql)
//   PORT       port to listen on (default: 4000)

import { readFileSync } from "node:fs";

import { ApolloServer } from "@apollo/server";
import { startStandaloneServer } from "@apollo/server/standalone";
import { Neo4jGraphQL } from "@neo4j/graphql";
import neo4j from "neo4j-driver";

const uri = process.env.NEO4J_URI ?? "bolt://localhost:7687";
const schemaPath = process.env.SCHEMA ?? "schema/sbml.graphql";
const port = Number(process.env.PORT ?? 4000);

// The database runs without authentication
const driver = neo4j.driver(uri);

// Wait for the database, which may still be starting (e.g. after `docker compose up`)
for (let attempt = 1; ; attempt++) {
  try {
    await driver.getServerInfo();
    break;
  } catch (error) {
    if (attempt === 30) throw error;
    console.log(`Waiting for ${uri} (${error.code ?? error.message})`);
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}
console.log(`Connected to ${uri}`);

const typeDefs = readFileSync(schemaPath, "utf-8");
const neoSchema = new Neo4jGraphQL({ typeDefs, driver });
const schema = await neoSchema.getSchema();
console.log(`Built GraphQL schema from ${schemaPath}`);

const server = new ApolloServer({ schema, introspection: true });
const { url } = await startStandaloneServer(server, { listen: { port } });
console.log(`GraphQL API ready at ${url}`);
