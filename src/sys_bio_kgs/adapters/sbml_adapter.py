import logging
from pathlib import Path
from typing import Iterator, Tuple, Dict, Any
import base64
from collections import defaultdict

from momapy.sbml.io import sbml

logger = logging.getLogger(__name__)


class SBMLAdapter:
    """
    BioCypher Adapter for SBML using momapy.

    Nodes (input_label):
      - entity                (SBML species)
      - process               (SBML reaction)
      - compartment           (SBML compartment)

    Edges (input_label):
      - reactant              (species → reaction)
      - product               (reaction → species)
      - modifier              (species → reaction)
      - contained entity      (species → compartment)
    """

    # --------------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------------
    def __init__(self, data_source: str | Path,
        annotations_as_node_properties: bool = True,
        **kwargs):
        """
        Args:
            data_source: Path to the SBML file
            **kwargs: Additional configuration parameters (unused for now)
        """
        self.sbml_path = Path(data_source)
        self.config = kwargs

        self.annotations_as_node_properties = annotations_as_node_properties

        logger.info(f"Loading SBML model with momapy: {self.sbml_path}")

        result = sbml.SBMLReader.read(self.sbml_path)
        self.model = result.obj
        self.annotations = result.annotations
        self.notes = result.notes

        logger.info("SBML model loaded successfully")

    # --------------------------------------------------------------
    # NODES
    # --------------------------------------------------------------
    def get_nodes(self) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
        """
        Yield nodes for BioCypher.

        Yields:
            (node_id, input_label, properties_dict)

        CURRENT SCHEMA:
            entity / process / compartment only have a `name` property.
        """

        # --- SBML model → model node ---
        yield (self.model.id_, "model", self.get_model_properties())


        # --- SBML compartments → compartment nodes ---
        for comp in self.model.compartments:
            props = {}
            node_id = comp.id_

            notes_base64 = self._parse_notes(self.notes.get(comp, None))
            if notes_base64 is not None:
                props["notes_base64"] = notes_base64
            if comp.name is not None:
                props["name"] = comp.name
            if self.annotations_as_node_properties:
                props.update(self._parse_annotations_to_node_properties(self.annotations.get(comp, None)))

            yield (node_id, "compartment", props)

        # --- SBML species → entity nodes ---
        for sp in self.model.species:
            props = {}
            node_id = sp.id_

            notes_base64 = self._parse_notes(self.notes.get(sp, None))
            if notes_base64 is not None:
                props["notes_base64"] = notes_base64
            if sp.name is not None:
                props["name"] = sp.name
            if self.annotations_as_node_properties:
                props.update(self._parse_annotations_to_node_properties(self.annotations.get(sp, None)))

            yield (node_id, "entity", props)

        # --- SBML reactions → process nodes ---
        for rx in self.model.reactions:
            props = {}
            node_id = rx.id_

            notes_base64 = self._parse_notes(self.notes.get(rx, None))
            if notes_base64 is not None:
                props["notes_base64"] = notes_base64
            if rx.name is not None:
                props["name"] = rx.name
            if self.annotations_as_node_properties:
                props.update(self._parse_annotations_to_node_properties(self.annotations.get(rx, None)))
            yield (node_id, "process", props)

    # --------------------------------------------------------------
    # EDGES
    # --------------------------------------------------------------
    def get_edges(self) -> Iterator[Tuple[str, str, str, str, Dict[str, Any]]]:
        """
        Yield edges for BioCypher.

        Yields:
            (edge_id, source_id, target_id, input_label, properties_dict)
        """

        # --- Reactant / Product / Modifier edges (species ↔ reaction) ---
        for rx in self.model.reactions:
            rx_id = rx.id_

            # Reactants: species → reaction
            for sr in rx.reactants:
                props = {}
                species_id = sr.referred_species.id_
                edge_id = f"{species_id}_reactant_{rx_id}"
                if sr.stoichiometry is not None:
                    props["stoichiometry"] = sr.stoichiometry
                yield (
                    edge_id,
                    species_id,
                    rx_id,
                    "reactant",
                    props,
                )

            # Products: reaction → species (aligned with SBGN process → product)
            for sr in rx.products:
                props = {}
                species_id = sr.referred_species.id_
                edge_id = f"{rx_id}_product_{species_id}"
                if sr.stoichiometry is not None:
                    props["stoichiometry"] = sr.stoichiometry
                yield (
                    edge_id,
                    rx_id,
                    species_id,
                    "product",
                    props,
                )

            # Modifiers: species → reaction
            for sr in rx.modifiers:
                species_id = sr.referred_species.id_
                edge_id = f"{species_id}_modifier_{rx_id}"
                props = {}  # can extend with SBO terms or roles later
                yield (
                    edge_id,
                    species_id,
                    rx_id,
                    "modifier",
                    props,
                )

        # --- Species localization: contained entity edges (species → compartment) ---
        for sp in self.model.species:
            if sp.compartment:
                species_id = sp.id_
                comp_id = sp.compartment.id_
                edge_id = f"{species_id}_contained_entity_{comp_id}"
                props: Dict[str, Any] = {}
                yield (
                    edge_id,
                    species_id,
                    comp_id,
                    "contained entity",
                    props,
                )


    # --------------------------------------------------------------
    # Node specific properties
    # --------------------------------------------------------------

    def get_model_properties(self):
        model = self.model
        properties = {
            "name": model.name,
            "notes_base64": self._parse_notes(self.notes.get(model, None)),
        }
        if self.annotations_as_node_properties:
            properties.update(self._parse_annotations_to_node_properties(self.annotations.get(model, None)))
        return properties

    # --------------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------------

    @staticmethod
    def _parse_notes(notes: frozenset) -> str:
        """Parse notes from SBML elements."""
        if notes is None:
            return None

        # encode notes in base64 for compatibility with neo4j CSV import
        for note in notes:
            break

        note = base64.b64encode(note, altchars=None).decode()
        return note

    @staticmethod
    def _parse_annotations_to_node_properties(annotations: frozenset) -> Dict[str, Any]:
        """Parse annotations from SBML elements."""
        if annotations is None:
            return {}

        parsed_annotations = defaultdict(list)
        for annotation in annotations:
            qualifier = annotation.qualifier.value
            for resource in annotation.resources:
                parsed_annotations[qualifier].append(resource)

        return parsed_annotations

    # --------------------------------------------------------------
    # METADATA
    # --------------------------------------------------------------
    def get_metadata(self) -> Dict[str, Any]:
        """
        Metadata for BioCypher / downstream tooling.
        """
        return {
            "name": "SBMLAdapter",
            "data_source": str(self.sbml_path),
            "data_type": "sbml",
            "version": "0.2.0",
            "adapter_class": "SBMLAdapter",
        }

    # --------------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------------
    def validate_data_source(self) -> bool:
        """
        Check if SBML file exists and is readable.

        Returns:
            True if valid, False otherwise.
        """
        if not self.sbml_path.exists():
            logger.error(f"SBML file not found: {self.sbml_path}")
            return False

        try:
            sbml.SBMLReader.read(self.sbml_path)
            return True
        except Exception as e:
            logger.error(f"Failed to parse SBML: {e}")
            return False
