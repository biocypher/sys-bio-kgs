# Export from Neo4j KG to XML

## Dependencies

neomodel
python-libsbml

Install with `pip install -e ".[export]"`.

The deploy container must be running (`docker-compose up`). Notes are
decoded from `notes_base64`, so the APOC decode step is not required.

## Usage

`python sbml_exporter.py --model_id 66b2d7a3-1ee3-47d9-b091-b2938e66b6fe --outfile output_1.xml`
