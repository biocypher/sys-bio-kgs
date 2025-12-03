#!/usr/bin/env python3

import logging
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from lxml import etree
from tqdm.auto import tqdm

DEFAULT_SBGN_DIR = "sbgn_annotated"
DEFAULT_SBML_DIR = "sbml"
DEFAULT_OUTPUT_CSV = "sbgn_sbml_identifier_overlap.csv"

SBGN_DIR = Path(os.environ.get("SBGN_ANNOTATED_DIR", DEFAULT_SBGN_DIR))
SBML_DIR = Path(os.environ.get("SBML_DIR", DEFAULT_SBML_DIR))
OUTPUT_CSV = Path(os.environ.get("IDENTIFIER_MATCH_CSV", DEFAULT_OUTPUT_CSV))

IDENTIFIER_PREFIXES = ("http://identifiers.org/", "https://identifiers.org/")
XML_PARSER = etree.XMLParser(remove_comments=True, recover=True)

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def list_files(directory: Path, suffixes: Tuple[str, ...]) -> List[Path]:
    """Return files in directory whose suffix matches suffixes (case-insensitive)."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    suffixes = tuple(s.lower() for s in suffixes)
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    logger.info("Discovered %d files in %s", len(files), directory)
    return files


def extract_identifiers(file_path: Path) -> Set[str]:
    """Parse XML file and collect identifiers.org URIs."""
    identifiers: Set[str] = set()
    try:
        tree = etree.parse(str(file_path), parser=XML_PARSER)
    except etree.XMLSyntaxError as exc:
        logger.warning("Failed to parse %s: %s", file_path, exc)
        return identifiers

    for elem in tree.iter():
        for attr_value in elem.attrib.values():
            if not attr_value:
                continue
            attr_str = str(attr_value).strip()
            if attr_str.startswith(IDENTIFIER_PREFIXES):
                identifiers.add(attr_str)
    return identifiers


def load_identifier_sets(directory: Path, suffixes: Tuple[str, ...]) -> Dict[str, Set[str]]:
    """Load all files under directory and map file name to identifiers set."""
    files = list_files(directory, suffixes)
    id_map: Dict[str, Set[str]] = {}
    for file_path in tqdm(files, desc=f"Parsing {directory.name}", unit="file"):
        id_map[file_path.name] = extract_identifiers(file_path)
    return id_map


def build_overlap_table(sbml_ids: Dict[str, Set[str]], sbgn_ids: Dict[str, Set[str]]) -> pd.DataFrame:
    """Compute pairwise overlaps between SBML and SBGN identifier sets."""
    sbgn_items = list(sbgn_ids.items())
    rows: List[Dict[str, object]] = []
    for sbml_file, sbml_set in tqdm(sbml_ids.items(), desc="Comparing SBML vs SBGN"):
        for sbgn_file, sbgn_set in sbgn_items:
            overlap = sorted(sbml_set & sbgn_set)
            rows.append(
                {
                    "sbml_file": sbml_file,
                    "sbgn_file": sbgn_file,
                    "sbml_identifier_count": len(sbml_set),
                    "sbgn_identifier_count": len(sbgn_set),
                    "overlap_count": len(overlap),
                    "overlap_urls": " ".join(overlap),
                }
            )
    df = pd.DataFrame(rows)
    return df


def main() -> None:
    """Entrypoint for computing identifier overlaps."""
    logger.info("Loading SBML identifier sets from %s", SBML_DIR)
    sbml_ids = load_identifier_sets(SBML_DIR, suffixes=(".xml", ".sbml"))
    logger.info("Loading SBGN identifier sets from %s", SBGN_DIR)
    sbgn_ids = load_identifier_sets(SBGN_DIR, suffixes=(".sbgn", ".xml"))

    logger.info(
        "Computing pairwise overlaps between %d SBML and %d SBGN files",
        len(sbml_ids),
        len(sbgn_ids),
    )
    overlap_df = build_overlap_table(sbml_ids, sbgn_ids)
    overlap_df.to_csv(OUTPUT_CSV, index=False)
    logger.info("Wrote %d pairwise records to %s", len(overlap_df), OUTPUT_CSV)


if __name__ == "__main__":
    main()
