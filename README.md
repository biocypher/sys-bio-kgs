# sys-bio-kgs

This repository originated from project 4 at the [4th BioHackathon Germany](https://www.denbi.de/de-nbi-events/1933-4th-biohackathon-germany-a-common-framework-for-transforming-systems-biology-models-into-human-and-ai-accessible-knowledge-graphs) (December 2025), hosted by de.NBI. The project brought together expertise in systems biology, knowledge management, and large language models (LLMs) to develop a common framework for transforming systems biology models into human- and AI-accessible knowledge graphs.

## BioHackathon Germany 2025
### Aims

[SBML](https://sbml.org/) and [SBGN-ML](https://sbgn.github.io/) are powerful, well-established standards for encoding biological models, but their XML-based format limits human readability, queryability, and interoperability. Knowledge graphs address these limitations by making complex biological relationships traversable, enrichable with external data sources (e.g. KEGG, OmniPath, Open Targets), and accessible to LLMs via RAG and MCP.

Leveraging the BioCypher ecosystem, our aim was to develop: 
- A common (and extendable) labelled property graph schema for system biology models, utilising as foundation existing standard ontologies (e.g. Biolink, SBO, EDAM, KiSAO),
- BioCypher adapters for SBGN and SBGN-ML, using the common schema,
- SBGN and SBGN-ML export functionalities, and
- One or more example applications, using either participant-provided use cases or models provided in BioModels (e.g. disease maps, metabolic maps, signalling networks, ODE models, Boolean models, GEMS).

<p align="center">
<img width="500" alt="biohackathon-project-approach" src="https://github.com/user-attachments/assets/cb858d73-f0d6-41aa-9d8a-bf6eda688473" />
</p>

### Results

We focused on the Repressilator model (a cyclic process of three proteins and their mRNAs), since it is available in well annotated form in both SBGN-ML and SBML formats. We used the [momapy](https://github.com/adrienrougny/momapy) library to support XML file ingestion. 


The workflow:
<p align="center">
<img width="2048" alt="workflow" src="https://github.com/user-attachments/assets/277dea41-3c50-43da-a78d-42ca40cf4058" />
</p>

The hackathon focused on the following tasks:

1. **Schema configuration** — defining a shared, extensible labelled property graph schema grounded in standard ontologies (Biolink, SBO, EDAM, KiSAO)  
Initial config based on SBO availible here: [config/simple_schema_config.yaml](config/simple_schema_config.yaml)

2. **SBGN BioCypher adapter** — transforming momapy objects into knowledge graph tuples
SBGN adapter, using the momapy library: [src/sys_bio_kgs/adapters/sbgn_adapter.py](src/sys_bio_kgs/adapters/sbgn_adapter.py)
   
3. **Extending [momapy](https://github.com/adrienrougny/momapy)** to support SBML parsing  
Implemeted in new branches:  
[momapy:sbml_kinetic](https://github.com/adrienrougny/momapy/tree/sbml_kinetic) for plain kinetic models in SBML  
[momapy:biohackathon_2025](https://github.com/adrienrougny/momapy/tree/biohackathon_2025) for GEM models in SBML
  
4. **SBML BioCypher adapter** — transforming momapy objects into knowledge graph tuples  
SBML adapter, using the updated momapy library: [src/sys_bio_kgs/adapters/sbml_adapter.py](src/sys_bio_kgs/adapters/sbml_adapter.py)

5. **KG-to-SBML export** — round-trip export from knowledge graph back to SBML format  
[export_scripts](export_scripts)  
Resulting round trip, showing the imported SBML and the exported SBML side-by-side:
<p align="center">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/52c01751-2dfc-458f-8878-e1475b10bb04" />
</p>

7. **Merging of Models** — finding the best strategy for linking model entities to KG nodes across heterogeneous sources  
Pairwise comparison of annotated SBGN and SBML files [sbgn_sbml_identifiers_match.py](scripts/sbgn_sbml_matching)  
Results on the comparison between SBML and SBGN model files seen in Neo4j:
<p align="center">
<img width="800" alt="image" src="https://github.com/user-attachments/assets/ef56f88f-3cf4-49a9-87d8-5135c4116934" />
</p>

8. **Benchmarking** — defining and evaluating the functions we expect the system to support
This was composed of two parts:  
  - User question curation: To test KG utility, we compiled a list of natural lanuguage questions from a survey sent to potential users. The questions are related to model content and structur. [data/user_questions.csv](data/user_questions.csv)  
  - Model curation: To develop test suites on the framework, models from Reactome and BioMoldes where compiles, and matching SBGN/SBML models annotated. [data/](data/)

## Repository overview

This BioCypher pipeline processes XML data using the available adapters to create a knowledge graph.

## Features

- **Data Source**: XML data processing
- **Adapter**: my_resource_adapter
- **Output**: Neo4j knowledge graph
- **Docker Support**: Containerized deployment
- **Testing**: Comprehensive test suite

## Installation

### Prerequisites

- Python 3.11 or higher
- Neo4j database (local or remote)

### Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd sys-bio-kgs
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

   Or using uv:
   ```bash
   uv sync
   ```

3. Configure your data source in `create_knowledge_graph.py`

4. Update the schema configuration in `config/schema_config.yaml` if needed

## Usage

### Basic Usage

Run the pipeline to create the knowledge graph:

```bash
python create_knowledge_graph.py
```

### Configuration

The pipeline uses two main configuration files:

- `config/biocypher_config.yaml` - BioCypher settings
- `config/schema_config.yaml` - Schema mapping configuration
### Docker Usage

The Docker pipeline has three steps. Each step reads from and writes to
directories in the repository, so steps can be run on their own and their
files inspected or deleted locally:

| Step | Command | Reads | Writes |
|---|---|---|---|
| build | `docker compose run --rm build` | `data/` | `biocypher-out/build2neo/`, `biocypher-log/` |
| import | `docker compose run --rm --no-deps import` | `biocypher-out/build2neo/` | `neo4j/data/` (overwritten) |
| deploy | `docker compose up -d --no-deps deploy` | `neo4j/data/` | `neo4j/logs/`, `neo4j/plugins/` |

`docker compose up -d` runs all three steps in order. Neo4j 5.26 Community (with APOC) is then
available at http://localhost:7474 (bolt: `localhost:7687`, no authentication);
stop it with `docker compose down`.

Create the host directories once before the first run; otherwise Docker
creates them owned by root:

```bash
mkdir -p biocypher-out/build2neo neo4j/{data,logs,plugins}
```

Notes:
- Containers run as your user (`UID`/`GID`, default `1000:1000`), so all files
  are owned by you. If your IDs differ, `export UID GID=$(id -g)` first.
- The build step can be replaced by running the pipeline on the host, which
  writes to the same place (`output_directory` in `config/biocypher_config.yaml`):
  `uv run python create_knowledge_graph_sbgn_sbml.py --clean`
- BioCypher adds to the output of earlier runs rather than replacing it, and
  the import loads all of it. `--clean` first removes the BioCypher files from
  earlier runs; leave it out to combine the output of several scripts.
  The build step passes `--clean` unless `CLEAN_OUTPUT=0` is set.
- `PIPELINE_SCRIPT=<script.py>` selects the script run by the build step
  (it must accept `--clean`, or use `CLEAN_OUTPUT=0`).
- `NEO4J_DIR=<dir>` keeps a separate database next to the default `neo4j/`,
  e.g. `NEO4J_DIR=neo4j_other docker compose run --rm --no-deps import`.
- The build step keeps its uv cache and virtualenv in `.cache/`.
- To start over: `rm -rf biocypher-out/build2neo/* neo4j/data`.

To export a model from the running database back to SBML, see
[export_scripts](export_scripts).

### BioChatter

[BioChatter Light](https://github.com/biocypher/biochatter-light) can answer
questions about the deployed graph in natural language (knowledge graph tab).
It reads the graph schema from the `Schema_info` node that the pipeline scripts
write (`write_schema_info`), and uses a Gemini model: put a key from
[Google AI Studio](https://aistudio.google.com) in `.env` (git-ignored) as
`GOOGLE_API_KEY=...`, then, with the database deployed:

```bash
docker compose --profile biochatter up -d --no-deps biochatter
```

and open http://localhost:8501. `BIOCHATTER_MODEL=<model>` selects the Gemini
model (default `gemini-flash-lite-latest`, which has usable free-tier rate
limits). The image only accepts keys for a fixed list of older Gemini models;
[docker/biochatter/sitecustomize.py](docker/biochatter/sitecustomize.py) adds
the configured model to that list at start-up.

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=sys_bio_kgs --cov-report=html
```

## Project Structure

```
sys-bio-kgs/
├── config/
│   ├── biocypher_config.yaml
│   └── schema_config.yaml
├── src/sys_bio_kgs/
│   └── adapters/
│       └── my_resource_adapter.py
├── create_knowledge_graph.py
├── docker-compose.yml
├── scripts/
│   ├── build.sh
│   └── import.sh
├── tests/
│   └── test_my_resource_adapter.py
├── pyproject.toml
└── README.md
```

## Development

### Code Style

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **mypy** for type checking

Format code:
```bash
black .
isort .
```

Type checking:
```bash
mypy src/
```

## License

MIT

## Author

Sebastian Lobentanzer - sebastian.lobentanzer@gmail.com
