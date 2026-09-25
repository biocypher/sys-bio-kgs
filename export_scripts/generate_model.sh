# Run when deploy is up

neomodel_inspect_database --db bolt://neo4j:password@localhost:7687 --write-to generated_models.py

# Reference:
# - https://neomodel.readthedocs.io/en/latest/getting_started.html#database-inspection-requires-apoc