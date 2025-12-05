import libsbml
from neomodel import db

from models import (
    Model,
    Process,
    PhysicalEntityRepresentation,
    PhysicalCompartment,
    ReactantRel,
    ProductRel,
    ModifierRel,
)


# ============================================================================
# Base Class
# ============================================================================

class SBMLWriter:
    """
    Base class providing common helper functionality for all SBML writer components.
    """

    def __init__(self, neomodel_model: Model, sbml_model: libsbml.Model):
        self.nm_model = neomodel_model     # the Neo4j Model node
        self.sbml_model = sbml_model       # the libSBML Model object

    @staticmethod
    def fail(msg):
        raise RuntimeError(msg)

    @staticmethod
    def to_sid(node, default_prefix):
        """
        Convert a neomodel node to a stable SBML id.
        """
        if hasattr(node, "id_") and node.id_:
            return node.id_
        return f"{default_prefix}_{node.element_id}"


# ============================================================================
# Compartment Writer
# ============================================================================

class SBMLCompartmentWriter(SBMLWriter):

    def write(self):
        """
        Export PhysicalCompartment → SBML <compartment>.
        """
        compartments = self.nm_model.compartment_of

        for comp in compartments:
            sid = self.to_sid(comp, "comp")

            c = self.sbml_model.createCompartment()
            c.setId(sid)
            c.setName(comp.name or sid)
            c.setConstant(True)
            c.setSBOTerm(comp.sbo or "SBO:0000290")


# ============================================================================
# Species Writer
# ============================================================================

class SBMLSpeciesWriter(SBMLWriter):

    def write(self):
        """
        Export PhysicalEntityRepresentation → SBML <species>.
        """
        species_nodes = self.nm_model.entity_of

        for node in species_nodes:
            sid = self.to_sid(node, "species")

            s = self.sbml_model.createSpecies()
            s.setId(sid)
            s.setName(node.name or sid)
            s.setSBOTerm(node.sbo or "SBO:0000245")

            # notes + annotations
            if node.notes:
                s.setNotes(node.notes)
            if node.notes_base64:
                s.appendNotes(f"<p>base64 notes: {node.notes_base64}</p>")

            # TODO: Add compartment assignment (if present)


# ============================================================================
# Reaction Writer
# ============================================================================

class SBMLReactionWriter(SBMLWriter):

    def write(self):
        """
        Export Process → SBML <reaction>.
        """
        processes = self.nm_model.process_of

        for proc in processes:
            rid = self.to_sid(proc, "rxn")

            r = self.sbml_model.createReaction()
            r.setId(rid)
            r.setName(proc.name or rid)
            r.setSBOTerm(proc.sbo or "SBO:0000375")

            self._write_reactants(proc, r)
            self._write_products(proc, r)
            self._write_modifiers(proc, r)

    # ---------- Subcomponents ----------
    def _write_reactants(self, proc, sbml_reaction):
        reactants = proc.reactant
        for species in reactants:
            rel = proc.reactant.relationship(species)
            sr = sbml_reaction.createReactant()
            sr.setSpecies(self.to_sid(species, "species"))
            if rel.stoichiometry is not None:
                sr.setStoichiometry(rel.stoichiometry)

    def _write_products(self, proc, sbml_reaction):
        products = proc.product
        for species in products:
            rel = proc.product.relationship(species)
            sp = sbml_reaction.createProduct()
            sp.setSpecies(self.to_sid(species, "species"))
            if rel.stoichiometry is not None:
                sp.setStoichiometry(rel.stoichiometry)

    def _write_modifiers(self, proc, sbml_reaction):
        modifiers = proc.modifier
        for species in modifiers:
            sm = sbml_reaction.createModifier()
            sm.setSpecies(self.to_sid(species, "species"))

# ============================================================================
# Main Exporter Class
# ============================================================================

class SBMLExporter:

    def __init__(self, model_id: str):
        try:
            self.model = Model.nodes.get(id_=model_id)
        except Model.DoesNotExist:
            raise RuntimeError(f"Model '{model_id}' not found")

        self.doc = None
        self.sbml_model = None

    def _create_document(self):
        self.doc = libsbml.SBMLDocument(3, 2)
        self.sbml_model = self.doc.createModel()

        self.sbml_model.setId(self.model.id_ or f"model_{self.model.element_id}")
        self.sbml_model.setName(self.model.name or self.sbml_model.getId())

        if self.model.sbo:
            self.sbml_model.setSBOTerm(self.model.sbo)

    def export(self, outfile: str):
        """
        Perform a full SBML export.
        """
        self._create_document()

        # --- Writers ---
        writers = [
            SBMLCompartmentWriter(self.model, self.sbml_model),
            SBMLSpeciesWriter(self.model, self.sbml_model),
            SBMLReactionWriter(self.model, self.sbml_model),
        ]

        for writer in writers:
            writer.write()

        # --- Write SBML ---
        result = libsbml.writeSBMLToFile(self.doc, outfile)
        if result != 1: # 1 (true) on success
            print(f"✘ SBML write failed with code {result}")
            raise RuntimeError("SBML write failed")

        print(f"✔ SBML written to {outfile}")


# ============================================================================
# Usage
# ============================================================================

def parse_args():
    """
    Defines and parses command-line arguments.
    Returns the populated argparse.Namespace object.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Export a Neo4j model to SBML using libSBML."
    )

    parser.add_argument(
        "--model_id",
        required=True,
        type=str,
        help="Value of the Model.id_ property to export."
    )

    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="Output SBML filename (default: <model_id>.xml)",
    )

    return parser.parse_args()


def main():

    from neomodel import get_config
    from models import Model

    config = get_config()
    config.database_url = 'bolt://neo4j:password@localhost:7687'  # default

    args = parse_args()

    model_id = args.model_id
    outfile = args.outfile or f"{model_id}.xml"

    exporter = SBMLExporter(model_id=model_id)
    exporter.export(outfile)

if __name__ == "__main__":
    main()

