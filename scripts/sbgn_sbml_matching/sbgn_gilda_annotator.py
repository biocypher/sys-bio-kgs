
"""Annotate SBGN glyphs using the INDRA grounding service."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import requests

SBGN_NS = "http://sbgn.org/libsbgn/0.3"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
BQMODEL_NS = "http://biomodels.net/model-qualifiers/"
GROUNDING_SERVICE_URL = "http://grounding.indra.bio/"
MIN_GROUNDING_SCORE = 0.75
ALL_GILDA_DATABASES = {
    "HGNC",
    "UP",
    "FPLX",
    "CHEBI",
    "GO",
    "DOID",
    "EFO",
    "HP",
    "MESH",
    "ADEFT",
}
DEFAULT_DATABASES = {"HGNC", "UP", "CHEBI"}
CLASS_DATABASE_HINTS: Dict[str, set[str]] = {
    "macromolecule": {"HGNC", "UP"},
    "simple chemical": {"CHEBI"},
}
SUPPORTED_SUFFIXES = (".sbgn", ".sbgnml", ".xml")

LOGGER = logging.getLogger("sbgn_grounding_annotator")
_GROUNDING_CACHE: Dict[Tuple[str, Tuple[str, ...]], Optional[Tuple[str, ...]]] = {}

ET.register_namespace("", SBGN_NS)
ET.register_namespace("rdf", RDF_NS)
ET.register_namespace("bqmodel", BQMODEL_NS)


def annotate_sbgn_files(
    sbgn_files: Iterable[Path],
    *,
    output_dir: Optional[Path] = None,
    min_score: float = MIN_GROUNDING_SCORE,
    grounding_service_url: str = GROUNDING_SERVICE_URL,
    use_default_naming: bool = False,
) -> List[Path]:
    """Annotate many SBGN files with optional bulk output handling.

    Args:
        sbgn_files: Paths to SBGN files that should be annotated.
        output_dir: Directory for annotated copies; defaults to in-place edits.
        min_score: Minimum acceptable grounding score.
        grounding_service_url: Grounding service base URL.
        use_default_naming: When True, rely on `_default_output_path` (ignored if
            `output_dir` is provided).

    Returns:
        List of paths to annotated files.
    """
    annotated_paths: List[Path] = []
    skipped: List[Path] = []
    destination_dir = Path(output_dir).expanduser().resolve() if output_dir else None
    if destination_dir:
        destination_dir.mkdir(parents=True, exist_ok=True)

    for sbgn_file in sbgn_files:
        sbgn_path = Path(sbgn_file)
        output_path: Optional[Path]
        if destination_dir:
            output_path = destination_dir / sbgn_path.name
        elif use_default_naming:
            output_path = None
        else:
            output_path = sbgn_path

        try:
            annotated_paths.append(
                annotate_sbgn_file(
                    sbgn_path,
                    output_path=output_path,
                    min_score=min_score,
                    grounding_service_url=grounding_service_url,
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            LOGGER.warning("%s", exc)
            skipped.append(sbgn_path)

    if skipped:
        LOGGER.warning(
            "Skipped %s file(s) that could not be annotated: %s",
            len(skipped),
            ", ".join(str(path) for path in skipped),
        )

    return annotated_paths


def annotate_sbgn_file(
    input_path: Path,
    *,
    output_path: Optional[Path] = None,
    min_score: float = MIN_GROUNDING_SCORE,
    grounding_service_url: str = GROUNDING_SERVICE_URL,
) -> Path:
    """Annotate a single SBGN file and write the annotated copy.

    Args:
        input_path: Source `.sbgn`, `.sbgnml`, or `.xml` file.
        output_path: Optional destination path; defaults to `_default_output_path`.
        min_score: Minimum acceptable grounding score.
        grounding_service_url: Grounding service base URL.

    Returns:
        Path to the annotated SBGN file.
    """
    resolved_input = Path(input_path).expanduser().resolve()
    if not resolved_input.exists():
        raise FileNotFoundError(f"Input file {resolved_input} does not exist.")

    tree = ET.parse(resolved_input)
    sbgn_ns = _detect_sbgn_namespace(tree.getroot())
    modified, eligible_found = _annotate_tree(
        tree,
        sbgn_ns=sbgn_ns,
        grounding_service_url=grounding_service_url,
        min_score=min_score,
    )

    if not eligible_found:
        raise ValueError(f"No eligible glyphs were found in {resolved_input}.")
    if not modified:
        raise ValueError(
            f"Eligible glyphs were present but no annotations met the criteria in {resolved_input}."
        )

    destination = (output_path or _default_output_path(resolved_input)).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if sbgn_ns:
        ET.register_namespace("", sbgn_ns)
    try:
        ET.indent(tree, space="  ", level=0)  # type: ignore[attr-defined]
    except AttributeError:
        # Python < 3.9 does not expose ET.indent; safe to skip formatting.
        pass
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    LOGGER.info("Annotated glyphs written to %s", destination)
    return destination


def annotate_sbgn_folder(
    folder_path: Path,
    *,
    output_dir: Optional[Path] = None,
    min_score: float = MIN_GROUNDING_SCORE,
    grounding_service_url: str = GROUNDING_SERVICE_URL,
) -> List[Path]:
    """Annotate all supported SBGN files within a folder.

    Args:
        folder_path: Directory containing `.sbgn`/`.sbgnml`/`.xml` files.
        output_dir: Directory where annotated copies should be stored. Defaults to
            writing `_annotated` files alongside the originals.
        min_score: Minimum acceptable grounding score.
        grounding_service_url: Grounding service base URL.

    Returns:
        Paths to successfully annotated files.
    """
    resolved_folder = Path(folder_path).expanduser().resolve()
    if not resolved_folder.is_dir():
        raise NotADirectoryError(f"{resolved_folder} is not a directory.")

    candidates = sorted(
        path for path in resolved_folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not candidates:
        raise ValueError(f"No SBGN files were found inside {resolved_folder}.")

    destination_dir = Path(output_dir).expanduser().resolve() if output_dir else None

    annotated_paths = annotate_sbgn_files(
        candidates,
        output_dir=destination_dir,
        min_score=min_score,
        grounding_service_url=grounding_service_url,
        use_default_naming=destination_dir is None,
    )
    if not annotated_paths:
        raise ValueError(f"No files in {resolved_folder} could be annotated successfully.")
    return annotated_paths


def _annotate_tree(
    tree: ET.ElementTree,
    *,
    sbgn_ns: Optional[str],
    grounding_service_url: str,
    min_score: float,
) -> Tuple[bool, bool]:
    """Annotate eligible glyphs inside a parsed SBGN document."""
    root = tree.getroot()
    namespaces = {"sbgn": sbgn_ns} if sbgn_ns else {}
    glyph_xpath = ".//sbgn:glyph" if sbgn_ns else ".//glyph"
    glyphs = root.findall(glyph_xpath, namespaces=namespaces or None)
    modified = False
    eligible_found = False

    for glyph in glyphs:
        glyph_class = (glyph.get("class") or "").strip().lower()
        expected_dbs = CLASS_DATABASE_HINTS.get(glyph_class, DEFAULT_DATABASES)
        if not expected_dbs:
            continue

        label_path = "sbgn:label" if sbgn_ns else "label"
        label = glyph.find(label_path, namespaces=namespaces or None)
        if label is None:
            continue

        text = (label.get("text") or "").strip()
        if not text:
            continue

        eligible_found = True
        resources = _ground_label(
            text,
            grounding_service_url=grounding_service_url,
            min_score=min_score,
            expected_dbs=expected_dbs,
        )
        if not resources:
            continue

        for resource in resources:
            if _attach_annotation(glyph, resource, sbgn_ns):
                modified = True

    return modified, eligible_found


def _ground_label(
    text: str,
    *,
    grounding_service_url: str,
    min_score: float,
    expected_dbs: Optional[Iterable[str]] = None,
) -> Optional[List[str]]:
    """Ground a glyph label with the configured remote service."""
    query = text.strip()
    if not query:
        return None

    normalized_dbs = tuple(sorted(db.upper() for db in (expected_dbs or DEFAULT_DATABASES)))
    allowed_dbs = set(normalized_dbs)
    cache_key = (query.lower(), normalized_dbs)
    if cache_key in _GROUNDING_CACHE:
        cached = _GROUNDING_CACHE[cache_key]
        return list(cached) if cached else None

    try:
        response = requests.post(
            urljoin(grounding_service_url, "ground"),
            json={"text": query},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.warning("Grounding request failed for %s: %s", query, exc)
        _GROUNDING_CACHE[cache_key] = None
        return None

    for candidate in response.json():
        try:
            score = float(candidate.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if score < min_score:
            continue

        term = candidate.get("term") or {}
        db = (term.get("db") or "").upper()
        if db not in ALL_GILDA_DATABASES:
            continue
        if allowed_dbs and db not in allowed_dbs:
            continue

        resources = _collect_resource_uris(candidate, allowed_dbs or DEFAULT_DATABASES)
        if not resources:
            continue

        resource_tuple = tuple(resources)
        _GROUNDING_CACHE[cache_key] = resource_tuple
        return resources

    _GROUNDING_CACHE[cache_key] = None
    return None


def _collect_resource_uris(candidate: Dict[str, Any], allowed_dbs: Iterable[str]) -> List[str]:
    """Return identifiers.org URIs from a Gilda candidate and its subsumed terms."""
    desired = {db.upper() for db in allowed_dbs}
    resources: List[str] = []

    def add_resource(db: Optional[str], term_id: Optional[str]) -> None:
        if not db or not term_id:
            return
        db_upper = db.upper()
        if db_upper not in desired:
            return
        resources.append(_format_resource_uri(db_upper, term_id))

    def harvest(term: Dict[str, Any]) -> None:
        if not term:
            return
        add_resource(term.get("db"), term.get("id"))
        add_resource(term.get("source_db"), term.get("source_id"))

    harvest(candidate.get("term") or {})
    for term in candidate.get("subsumed_terms") or []:
        harvest(term or {})

    deduped: List[str] = []
    seen: set[str] = set()
    for resource in resources:
        if resource in seen:
            continue
        seen.add(resource)
        deduped.append(resource)
    return deduped


def _attach_annotation(glyph: ET.Element, resource_uri: str, sbgn_ns: Optional[str]) -> bool:
    """Attach RDF annotations to a glyph."""
    glyph_id = glyph.get("id")
    if not glyph_id:
        return False

    extension = glyph.find(_qname(sbgn_ns, "extension"))
    if extension is None:
        extension = ET.SubElement(glyph, _qname(sbgn_ns, "extension"))

    annotation = extension.find(_qname(sbgn_ns, "annotation"))
    if annotation is None:
        annotation = ET.SubElement(extension, _qname(sbgn_ns, "annotation"))

    rdf = annotation.find(f"./{{{RDF_NS}}}RDF")
    if rdf is None:
        rdf = ET.SubElement(annotation, f"{{{RDF_NS}}}RDF")

    description = None
    for candidate in rdf.findall(f"./{{{RDF_NS}}}Description"):
        about_value = candidate.attrib.get(f"{{{RDF_NS}}}about")
        if about_value in (glyph_id, f"#{glyph_id}"):
            description = candidate
            break
    if description is None:
        description = ET.SubElement(
            rdf,
            f"{{{RDF_NS}}}Description",
            attrib={f"{{{RDF_NS}}}about": f"#{glyph_id}"},
        )

    bqmodel_is = description.find(f"./{{{BQMODEL_NS}}}is")
    if bqmodel_is is None:
        bqmodel_is = ET.SubElement(description, f"{{{BQMODEL_NS}}}is")

    bag = bqmodel_is.find(f"./{{{RDF_NS}}}Bag")
    if bag is None:
        bag = ET.SubElement(bqmodel_is, f"{{{RDF_NS}}}Bag")

    for li in bag.findall(f"./{{{RDF_NS}}}li"):
        if li.attrib.get(f"{{{RDF_NS}}}resource") == resource_uri:
            return False

    ET.SubElement(bag, f"{{{RDF_NS}}}li", attrib={f"{{{RDF_NS}}}resource": resource_uri})
    return True


def _format_resource_uri(db: str, term_id: str) -> str:
    """Return a canonical identifiers.org URI for the provided term."""
    upper_db = db.upper()
    lowercase_prefix = {
        "UP": "uniprot",
    }.get(upper_db, upper_db.lower())

    keep_prefix = upper_db in {"CHEBI"}
    accession = term_id
    if not keep_prefix and term_id.upper().startswith(f"{upper_db}:"):
        accession = term_id.split(":", 1)[1]

    return f"http://identifiers.org/{lowercase_prefix}/{accession}"


def _detect_sbgn_namespace(root: ET.Element) -> Optional[str]:
    """Detect the namespace used by the SBGN document."""
    tag = root.tag
    if tag.startswith("{") and "}" in tag:
        return tag[1 : tag.index("}")]
    return None


def _qname(namespace: Optional[str], tag: str) -> str:
    """Return an ElementTree QName string for the provided namespace/tag."""
    return f"{{{namespace}}}{tag}" if namespace else tag


def _default_output_path(input_path: Path) -> Path:
    """Return `<name>_annotated` with the original extension preserved.

    Args:
        input_path: Path of the source file to annotate.

    Returns:
        Path pointing to the default annotated filename next to the source.
    """
    suffix = input_path.suffix
    if suffix:
        return input_path.with_name(f"{input_path.stem}_annotated{suffix}")
    return input_path.with_name(f"{input_path.name}_annotated")


def parse_args() -> argparse.Namespace:
    """Create and validate CLI arguments.

    Returns:
        Parsed command-line namespace containing runtime configuration.
    """
    parser = argparse.ArgumentParser(
        description="Annotate SBGN glyphs using the INDRA grounding service.",
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "-i",
        "--input",
        type=Path,
        help="Path to a single SBGN file to annotate.",
    )
    target_group.add_argument(
        "-f",
        "--folder",
        type=Path,
        help="Path to a folder containing SBGN files to annotate.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output location: file path when using --input, directory when using --folder "
            "(defaults to adding '_annotated' next to each source)."
        ),
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_GROUNDING_SCORE,
        help=f"Minimum acceptable grounding score (default: {MIN_GROUNDING_SCORE}).",
    )
    args = parser.parse_args()
    if args.folder and args.output and args.output.exists() and not args.output.is_dir():
        parser.error("--output must be a directory when annotating a folder.")
    if args.input and args.output and args.output.exists() and args.output.is_dir():
        parser.error("--output must be a file path when annotating a single file.")
    return args


def main() -> int:
    """CLI entry point.

    Returns:
        Integer exit status following conventional shell semantics.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    try:
        if args.folder:
            annotated = annotate_sbgn_folder(
                args.folder,
                output_dir=args.output,
                min_score=args.min_score,
                grounding_service_url=GROUNDING_SERVICE_URL,
            )
            LOGGER.info("Annotated %s file(s) inside %s.", len(annotated), args.folder)
        else:
            destination = annotate_sbgn_file(
                args.input,
                output_path=args.output,
                min_score=args.min_score,
                grounding_service_url=GROUNDING_SERVICE_URL,
            )
            LOGGER.info("Annotated file written to %s.", destination)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        LOGGER.error(str(exc))
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
