"""
Benchmark natural-language questions against the deployed knowledge graph.

For each in-scope case in questions.yaml, a Cypher query is generated (with
BioChatter's BioCypherPromptEngine and an LLM), run against Neo4j, and the result
is compared with the expected values. With --model reference, the reference
queries from questions.yaml are run instead (to check the question set itself).

Usage (from the repository root, with the SBML graph deployed):

    # check the question set
    uv run --no-project --with "neo4j<5" --with pyyaml \
        python benchmark/run_benchmark.py --model reference

    # benchmark a Gemini model (GOOGLE_API_KEY from the environment or .env)
    uv run --no-project --with "biochatter==0.14.2" --with langchain-google-genai --with pyyaml \
        python benchmark/run_benchmark.py --model gemini-flash-latest

Results are written to benchmark/results/<model>.csv.
"""

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import neo4j
import yaml

HERE = Path(__file__).parent
WRITE_CLAUSES = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|FOREACH)\b", re.I)


def load_api_key(name: str) -> str | None:
    """Read an API key from the environment or the repository's .env file."""
    if os.getenv(name):
        return os.environ[name]
    env_file = HERE.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    return None


def get_schema_info(driver) -> dict:
    """Schema info written by BioCypher's write_schema_info(as_node=True)."""
    with driver.session() as session:
        record = session.run("MATCH (n:Schema_info) RETURN n.schema_info AS s LIMIT 1").single()
    if record is None:
        raise SystemExit("No Schema_info node in the database: rebuild the graph with write_schema_info")
    return json.loads(record["s"])


def make_query_generator(model: str, schema_info: dict):
    """Return a function question -> Cypher query, using BioChatter and a Gemini model."""
    from biochatter.llm_connect import LangChainConversation
    from biochatter.prompts import BioCypherPromptEngine

    api_key = load_api_key("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not set (environment or .env)")
    # LangChain's Google connector reads the key from the environment;
    # BioChatter's set_api_key() does not pass it on
    os.environ["GOOGLE_API_KEY"] = api_key

    def conversation_factory():
        conversation = LangChainConversation(
            model_provider="google_genai", model_name=model, prompts={}, correct=False
        )
        if not conversation.set_api_key(api_key=api_key, user="benchmark"):
            raise SystemExit(f"Could not connect to {model} (google_genai)")
        return conversation

    engine = BioCypherPromptEngine(
        schema_config_or_info_dict=schema_info, conversation_factory=conversation_factory
    )
    return lambda question: engine.generate_query(question=question, query_language="Cypher")


def clean_query(query: str) -> str:
    """Strip Markdown code fences that LLMs often add."""
    query = query.strip()
    match = re.search(r"```(?:cypher)?\s*(.*?)```", query, re.S | re.I)
    return (match.group(1) if match else query).strip()


def flatten(value) -> list:
    """All scalar values in a query result (lists, maps and nodes are expanded)."""
    if isinstance(value, (list, tuple)):
        return [v for item in value for v in flatten(item)]
    if isinstance(value, dict):
        return [v for item in value.values() for v in flatten(item)]
    if hasattr(value, "items") and hasattr(value, "labels"):  # neo4j Node
        return [v for item in dict(value.items()).values() for v in flatten(item)]
    return [value]


def normalise(value) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip().lower()


def score(records: list, expected: dict) -> tuple[float, bool]:
    """Recall of the expected values in the result, and whether the answer is correct
    (all expected values present and, if given, the expected number of rows)."""
    returned = {normalise(v) for record in records for v in flatten(list(record.values()))}
    values = expected["values"]
    found = 0
    for item in values:
        alternatives = item if isinstance(item, list) else [item]
        found += any(normalise(a) in returned for a in alternatives)
    recall = found / len(values)
    rows_ok = expected.get("rows") is None or expected["rows"] == len(records)
    return recall, recall == 1 and rows_ok


def parts_score(query: str, parts: list) -> str:
    """BioChatter-style score: regular expressions found in the query."""
    hits = sum(re.search(part, query) is not None for part in parts)
    return f"{hits}/{len(parts)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="reference", help="'reference' or a Gemini model name")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--questions", default=HERE / "questions.yaml", type=Path)
    parser.add_argument("--repeats", default=1, type=int, help="runs per case (LLM output varies)")
    parser.add_argument("--sleep", default=0.0, type=float, help="seconds between cases (rate limits)")
    args = parser.parse_args()

    cases = yaml.safe_load(args.questions.read_text())["in_scope"]
    driver = neo4j.GraphDatabase.driver(args.uri)

    if args.model == "reference":
        generate = None
    else:
        generate = make_query_generator(args.model, get_schema_info(driver))

    rows = []
    for case in cases:
        for repeat in range(1, args.repeats + 1):
            started = time.time()
            query, error, records = case["cypher"], "", []
            try:
                if generate:
                    query = clean_query(generate(case["prompt"]))
                if WRITE_CLAUSES.search(query):
                    raise ValueError("refused: query modifies the database")
                with driver.session(default_access_mode=neo4j.READ_ACCESS) as session:
                    records = [record.data() for record in session.run(query)]
            except Exception as e:  # noqa: BLE001 - report any failure as a result
                error = f"{type(e).__name__}: {e}".replace("\n", " ")[:300]
            recall, correct = score(records, case["expected"]) if not error else (0.0, False)
            rows.append({
                "case": case["case"],
                "source": case["source"],
                "category": case["category"],
                "repeat": repeat,
                "model": args.model,
                "correct": correct,
                "recall": round(recall, 2),
                "rows": len(records),
                "parts_of_query": parts_score(query, case.get("parts_of_query", [])),
                "seconds": round(time.time() - started, 1),
                "query": " ".join(query.split()),
                "error": error,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            mark = "ok " if correct else "ERR" if error else "no "
            print(f"{mark} {case['case']:26} recall={recall:.2f} rows={len(records)} {error[:80]}")
            if args.sleep:
                time.sleep(args.sleep)
    driver.close()

    out = HERE / "results" / f"{args.model}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    n_correct = sum(r["correct"] for r in rows)
    print(f"\n{args.model}: {n_correct}/{len(rows)} correct -> {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
