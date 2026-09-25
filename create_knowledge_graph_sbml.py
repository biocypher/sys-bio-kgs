#!/usr/bin/env python3
"""
sys-bio-kgs - A repository for the implementations of the 2025 BioHackathon Germany that do not have another home already

This script creates a knowledge graph from an SBML model only, using BioCypher and the SBMLAdapter.
"""

import argparse
import logging

from biocypher import BioCypher

from sys_bio_kgs.adapters.sbml_adapter import SBMLAdapter
from sys_bio_kgs.output import clean_biocypher_output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove BioCypher output from earlier runs before writing. "
        "Omit to add to existing output, e.g. when building from several scripts.",
    )
    parser.add_argument(
        "--sbml",
        default="data/matched_annotated_repressilator_BIOMD0000000012.xml",
        help="SBML file to load",
    )
    return parser.parse_args()


def main():
    """Main function to create the knowledge graph."""
    args = parse_args()
    logger.info("Starting sys-bio-kgs SBML knowledge graph creation")

    bc = BioCypher(
        biocypher_config_path="config/biocypher_config.yaml",
        schema_config_path="config/simple_schema_config.yaml",
    )

    if args.clean:
        # output directory as resolved by BioCypher from the config
        clean_biocypher_output(bc._output_directory)

    adapter = SBMLAdapter(data_source=args.sbml)

    logger.info("Creating SBML knowledge graph...")
    bc.write_nodes(adapter.get_nodes())
    bc.write_edges(adapter.get_edges())
    logger.info("SBML knowledge graph creation completed successfully!")

    bc.write_import_call()
    bc.summary()


if __name__ == "__main__":
    main()
