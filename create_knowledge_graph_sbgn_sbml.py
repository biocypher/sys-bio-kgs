#!/usr/bin/env python3
"""
sys-bio-kgs - A repository for the implementations of the 2025 BioHackathon Germany that do not have another home already

This script creates a knowledge graph using BioCypher and the SBGNAdapter.
"""

import logging
from pathlib import Path

from biocypher import BioCypher

from sys_bio_kgs.adapters.sbml_adapter import SBMLAdapter as Adapter
from sys_bio_kgs.adapters.momapy_sbgn_adapter import MoMaPySBGNAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to create the knowledge graph."""
    logger.info("Starting sys-bio-kgs knowledge graph creation")
    
    # Initialize BioCypher
    bc = BioCypher(
        biocypher_config_path="config/biocypher_config.yaml",
        schema_config_path="config/simple_schema_config.yaml",
    )
    
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
    bc.summary()


if __name__ == "__main__":
    main()
