# Question benchmark

Natural-language questions about the knowledge graph, with reference Cypher
queries and expected answers, to test how well an LLM-based tool answers them
from the graph. The questions come from the user survey in
[`data/user_questions.csv`](../data/user_questions.csv); the placeholders there
(`[element]`, `[compartment]`, ...) are filled in for the SBML repressilator
graph.

The question set itself does not depend on a particular tool. `run_benchmark.py`
currently generates the queries with [BioChatter](https://github.com/biocypher/biochatter)
(`BioCypherPromptEngine`, the same approach as the knowledge graph tab of
BioChatter Light) and a Gemini model.

| File | Purpose |
|---|---|
| `questions.yaml` | The questions: `in_scope` (answerable, with reference query and expected values) and `later` (not answerable yet, with the reason) |
| `run_benchmark.py` | Generates a query per in-scope question, runs it against Neo4j and scores the result |
| `results/<model>.csv` | Results per model: score, generated query, errors |

## Questions

All 20 survey questions are included. Four are split into a part that the graph
can answer and a part that it cannot, giving 24 cases:

- **in_scope (9)**: model description, regulators of transcription and of an
  element, compartments and their contents, the reactions of an element or
  between two elements, model size and the most connected species.
- **later (15)**: questions that need data the graph does not have yet
  (modifier roles, kinetic laws and parameters: in the SBML file but not imported
  by the SBML adapter; reaction-level literature, expression data), the combined
  SBGN + SBML graph, a disease context, or are out of scope for a model graph.

## Scoring

Each generated query is run against the database (queries that would modify it
are refused) and compared with `expected` in `questions.yaml`:

- `values`: all must appear somewhere in the result (any column, any row; lists,
  maps and nodes are expanded). A nested list means "any of", e.g. `[6, 7]`.
- `rows`: the number of result rows, where it matters (e.g. listing six species
  and nothing else); `null` means not checked.

A case is **correct** if all values are found and the row count matches. The
results also report the recall of the expected values and, for comparison with
the [BioChatter benchmark](https://biochatter.org/benchmarking/), how many of the
`parts_of_query` regular expressions the query contains (not used for the score).

## Run

Deploy the SBML graph first; the pipeline writes the schema information that
BioChatter needs into the graph (`Schema_info` node):

```bash
PIPELINE_SCRIPT=create_knowledge_graph_sbml.py docker compose up -d
```

Check the question set by running the reference queries (all should be correct):

```bash
uv run --no-project --with "neo4j<5" --with pyyaml \
    python benchmark/run_benchmark.py --model reference
```

Benchmark a Gemini model. The key is read from `GOOGLE_API_KEY` in the
environment or in `.env` (git-ignored); keys are available from
[Google AI Studio](https://aistudio.google.com):

```bash
uv run --no-project --python 3.11 --with "biochatter==0.14.2" --with langchain-google-genai --with pyyaml \
    python benchmark/run_benchmark.py --model gemini-flash-lite-latest --sleep 5
```

The benchmark runs in its own environment (`uv run --no-project`): BioChatter
requires the Neo4j Python driver < 5, the project's `export` extra (neomodel)
requires >= 5. The 4.4 driver works with the Neo4j 5.26 server.

Options: `--repeats N` runs each case N times (LLM output varies), `--sleep S`
pauses between cases for rate limits. One question takes about four LLM requests
(entity, relationship and property selection, then the query). On the free tier,
`gemini-flash-latest` allows only 5 requests per minute, so use `--sleep 60`;
`gemini-flash-lite-latest` has higher limits.

## Results

| Model | Correct | Notes |
|---|---|---|
| reference queries | 9/9 | checks the question set |
| `gemini-flash-lite-latest` (gemini-3.5-flash-lite) | 4/9 | 2026-09-26, one run |

Observations so far:

- The annotation properties (`hasPart`, `is`, `isVersionOf`, ... from the
  BioModels qualifiers) read like relationships, and models use them as such,
  e.g. `(m:Model)-[:IsEntityOf|hasPart|is*0..]->(p)`. Renaming them (e.g.
  `annotation_hasPart`) may help.
- BioChatter passes edge directions from the schema's `source`/`target`, written
  as `(:Source)-(:Relationship)->(:Target)` (round brackets, not Cypher's `[ ]`);
  weaker models sometimes still reverse edges.
- "Species" has no class of its own: species are `physical entity representation`,
  which the compartment is a subclass of, so counting species may include it
  (accepted in `model_size`, not in `most_connected`).

## Adding questions

Add a case to `in_scope` with `prompt`, `cypher` and `expected`, run
`--model reference` to check it, or to `later` with the `reason` it cannot be
answered yet. The format follows the
[BioChatter benchmark data](https://github.com/biocypher/biochatter/tree/main/benchmark/data)
(`case`, `input`/`prompt`, `expected`, `parts_of_query`), so cases can be
adapted for it.
