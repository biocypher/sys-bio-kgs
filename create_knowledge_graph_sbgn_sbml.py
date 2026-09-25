#!/usr/bin/env python3
"""
sys-bio-kgs - A repository for the implementations of the 2025 BioHackathon Germany that do not have another home already

This script creates a knowledge graph using BioCypher and the SBGNAdapter.
"""

import argparse
import logging
from pathlib import Path

from biocypher import BioCypher

from sys_bio_kgs.adapters.sbml_adapter import SBMLAdapter as Adapter
from sys_bio_kgs.adapters.momapy_sbgn_adapter import MoMaPySBGNAdapter
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
    return parser.parse_args()


def main():
    """Main function to create the knowledge graph."""
    args = parse_args()
    logger.info("Starting sys-bio-kgs knowledge graph creation")
    
    # Initialize BioCypher
    bc = BioCypher(
        biocypher_config_path="config/biocypher_config.yaml",
        schema_config_path="config/simple_schema_config.yaml",
    )

    if args.clean:
        # output directory as resolved by BioCypher from the config
        clean_biocypher_output(bc._output_directory)
    
    # Initialize the SBGN adapter
    sbml_data_source = "data/matched_annotated_repressilator_BIOMD0000000012.xml"
    
    adapter = Adapter(
        data_source=sbml_data_source,
        # Add any additional configuration parameters here
    )
    
    # Create the knowledge graph
    logger.info("Creating SBML knowledge graph...")
    bc.write_nodes(adapter.get_nodes())
    try:
        bc.write_edges(adapter.get_edges())
    except StopIteration:
        logger.warning("No edges found to write to the knowledge graph.")

    logger.info("SBML knowledge graph creation completed successfully!")

    # Initialize the SBGN adapter
    sbgn_data_source = "data/matched_annotated_Repressilator_PD_v7.sbgn"

    adapter = MoMaPySBGNAdapter(
        data_source=sbgn_data_source,
        # Add any additional configuration parameters here
    )

    # Create the knowledge graph
    logger.info("Creating SBGN knowledge graph...")
    bc.write_nodes(adapter.get_nodes())
    try:
        bc.write_edges(adapter.get_edges())
    except StopIteration:
        logger.warning("No edges found to write to the knowledge graph.")

    logger.info("SBGN knowledge graph creation completed successfully!")

    # Create import script and final summary
    bc.write_import_call()
    # Schema info as a node in the graph, e.g. for BioChatter's knowledge graph tab
    bc.write_schema_info(as_node=True)
    bc.summary()


if __name__ == "__main__":
    main()
