import logging
from pathlib import Path
from typing import Iterator, Tuple, Dict, Any

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
    def __init__(self, data_source: str | Path, **kwargs):
        """
        Args:
            data_source: Path to the SBML file
            **kwargs: Additional configuration parameters (unused for now)
        """
        self.sbml_path = Path(data_source)
        self.config = kwargs

        logger.info(f"Loading SBML model with momapy: {self.sbml_path}")

        result = sbml.SBMLReader.read(self.sbml_path)
        self.model = result.obj
        self.annotations = result.annotations

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
            node_id = comp.id_
            props: Dict[str, Any] = {
                "name": comp.name or "",
                "notes_base64": self._parse_notes(self.notes.get(comp, None)),





            }
            props.update(self._parse_annotations(self.annotations.get(comp, None)))

            yield (node_id, "compartment", props)

        # --- SBML species → entity nodes ---
        for sp in self.model.species:
            node_id = sp.id_
            props = {
                "name": sp.name or "",
                "notes_base64": self._parse_notes(self.notes.get(sp, None)),





            }
            props.update(self._parse_annotations(self.annotations.get(sp, None)))

            yield (node_id, "entity", props)

        # --- SBML reactions → process nodes ---
        for rx in self.model.reactions:
            node_id = rx.id_
            props = {
                "name": rx.name or "",
                "notes_base64": self._parse_notes(self.notes.get(rx, None)),







            }
            props.update(self._parse_annotations(self.annotations.get(rx, None)))

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
                species_id = sr.referred_species.id_
                edge_id = f"{species_id}_reactant_{rx_id}"
                props: Dict[str, Any] = {

                    "stoichiometry": 1.0 if sr.stoichiometry is None else sr.stoichiometry







                }
                yield (
                    edge_id,
                    species_id,
                    rx_id,
                    "reactant",
                    props,
                )

            # Products: reaction → species (aligned with SBGN process → product)
            for sr in rx.products:
                species_id = sr.referred_species.id_
                edge_id = f"{rx_id}_product_{species_id}"
                props = {

                    "stoichiometry": 1.0 if sr.stoichiometry is None else sr.stoichiometry





                }
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
                props = {





                }  # can extend with SBO terms or roles later
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
                props: Dict[str, Any] = {






                }
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
        }
        return properties

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
