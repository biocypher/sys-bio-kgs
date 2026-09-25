"""Helpers for the BioCypher output directory."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Files written by the BioCypher neo4j batch writer
BIOCYPHER_OUTPUT_PATTERNS = (
    "*-header.csv",
    "*-part*.csv",
    "*-part*.parquet",
    "neo4j-admin-import-call.sh",
)


def clean_biocypher_output(output_directory: str | Path | None) -> int:
    """
    Remove BioCypher output from earlier runs in `output_directory`.

    BioCypher adds new part files next to existing ones, and the generated
    import call loads all of them, so output from earlier runs would be
    imported again. Only files matching BIOCYPHER_OUTPUT_PATTERNS are removed.

    Returns the number of files removed.
    """
    if output_directory is None:
        return 0
    output_directory = Path(output_directory)
    if not output_directory.is_dir():
        return 0

    removed = 0
    for pattern in BIOCYPHER_OUTPUT_PATTERNS:
        for path in output_directory.glob(pattern):
            path.unlink()
            removed += 1
    logger.info(f"Removed {removed} files from earlier runs in {output_directory}")
    return removed
