#!/bin/bash
# Import step: load the BioCypher output (mounted at /build2neo) into /data
# with neo4j-admin import. Overwrites the existing database.
set -euo pipefail

IMPORT_CALL=/build2neo/neo4j-admin-import-call.sh
if [ ! -f "$IMPORT_CALL" ]; then
  echo "$IMPORT_CALL not found: run the build step first" >&2
  exit 1
fi
bash "$IMPORT_CALL"

# Start once so the database is initialised, then stop
neo4j start
sleep 10
neo4j stop
