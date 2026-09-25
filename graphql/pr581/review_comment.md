Thanks for picking this up, @iqb430, and sorry for the slow reply!

I tested the PR (at `852dae1`) on a real BioCypher graph: an SBML model
(the repressilator) built with BioCypher 0.17 into Neo4j 5.26, serving the
generated schema with the current Neo4j GraphQL Library (`@neo4j/graphql` 7.6.3)
and comparing query results with Cypher. I did the testing and drafted this
review together with Claude Code (Anthropic's AI coding assistant).
Setup, generated schemas, test scripts and full findings (with steps to
reproduce): https://github.com/biocypher/sys-bio-kgs/tree/3dbd9e9/graphql/pr581

In short: the generated schema can't be used as is. Most of the problems come
from one design choice, so I've started with that.

### Main point: use BioCypher's internals instead of re-implementing them

The generator reads the raw schema dict and derives names with its own
`_pascal_case` / `_screaming_snake_case`. But the point of #500 is that the
GraphQL schema matches what BioCypher actually writes to Neo4j, and BioCypher
already has one naming pipeline for that: every label and relationship type is
`Translator.name_sentence_to_pascal(parse_label(name))`. The conversions differ
for relationship types, and for node labels as well:

| Schema name | BioCypher writes | PR node type / label | PR relationship type |
|---|---|---|---|
| contained entity | `ContainedEntity` | `ContainedEntity` | `CONTAINED_ENTITY` ❌ |
| mRNA | `MRNA` | `Mrna` ❌ | `MRNA` |
| DNA sequence variant | `DNASequenceVariant` | `DnaSequenceVariant` ❌ | `DNA_SEQUENCE_VARIANT` ❌ |
| non-coding RNA | `NoncodingRNA` | `NonCodingRna` ❌ | `NON_CODING_RNA` ❌ |
| SBML model | `SBMLModel` | `SbmlModel` ❌ | `SBML_MODEL` ❌ |

(The relationship type in my example in #500 was illustrative only.) In the test, every relationship query returned 0 results
(Cypher: 6 each) until the types were renamed.

Working from the dict also misses what BioCypher resolves in the ontology:
`is_a` and the hierarchy (the main reason for #500 over Neo4j's introspector),
`inherit_properties` / `exclude_properties`, `label_as_edge` / `synonym_for`,
and tail ontologies. The NetworkX metagraph writer in #455 shows a pattern that
could work here: implement it as a writer (or a `BioCypher` method) that gets the
`Translator`, take names from `translator.mappings` / `ontology.get_renaming()`,
the hierarchy from the ontology's NetworkX graph, and properties from
`ontology.mapping.extended_schema`. #455 also records edge source/target types
from the data, so it doesn't depend on `source`/`target` being in the schema.
Both would build on how #435 / #516 settle edge renaming.

### Other issues found in testing

1. **Not valid for the current Neo4j GraphQL Library.** `interface ... @relationshipProperties`
   is rejected (`Directive "@relationshipProperties" may not be used on INTERFACE`);
   since v4 this is `type ... @relationshipProperties`.
2. **Edges without `source`/`target` are dropped without a warning**: only an unused
   `...Props` type is generated. `source`/`target` are optional in BioCypher
   schemas, so this should at least warn (or use the data, as above).
3. **Only outgoing relationships.** A reaction can't list its reactants
   (`Cannot query field "reactant" on type "Process"`). Each relationship
   needs a field on both types (`direction: OUT` on one, `IN` on the other).
4. **Array properties become `String`.** `str[]` properties (e.g. annotation
   lists) fail at query time with `String cannot represent value: ["http://identifiers.org/uniprot:P03023"]`;
   they should be `[String!]` (likewise for `int[]`, `float[]`, ...). Unknown types
   silently default to `String`; a warning would help.
5. **Mutations are exposed.** The schema generates create/update/delete
   operations. For exposing a knowledge graph, read-only
   (`extend schema @mutation(operations: [])`) seems the better default,
   maybe with an option.
6. **Field names** like `containedentity` / `isentityof` are hard to read;
   camelCase (`containedEntity`) would be a start. Names that describe the
   role (`reactants`, `products`, `compartment`) would be even nicer, but that
   may need a schema option.
7. It isn't wired into the API yet, and there are no tests or docs.

One thing a generator will need to handle: BioCypher gives nodes the labels
of their parent classes too, and the Neo4j GraphQL Library matches nodes on
labels they have. So a type for a parent class also returns subclass nodes.
For example, the compartment appears among the species, and Neo4j's own
introspector has the same issue. How best to handle this, e.g. with GraphQL
interfaces for parent classes, is still an open question.

With fixes for 1 and 4 and the renamed relationship types, all relationship
and property queries matched Cypher, so the overall approach works. For reference, here is an excerpt
of the hand-written schema I compared against:

```graphql
extend schema @mutation(operations: [])

type Process implements SystemsBiologyRepresentation @node(labels: ["Process"]) {
  id: ID!
  name: String
  isVersionOf: [String!]
  reactants: [PhysicalEntityRepresentation!]! @relationship(type: "Reactant", direction: IN, properties: "ReactantProperties")
  products: [PhysicalEntityRepresentation!]! @relationship(type: "Product", direction: OUT, properties: "ProductProperties")
}

type ReactantProperties @relationshipProperties {
  stoichiometry: Float
}
```

Happy to test again against the same setup when there's a new version.
