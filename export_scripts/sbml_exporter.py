import base64
import re

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
        Convert a neomodel node to a stable, valid SBML id.

        SIds must match [A-Za-z_][A-Za-z0-9_]*, so e.g. UUIDs are rewritten
        (hyphens -> underscores, leading digit -> prefixed).
        """
        if hasattr(node, "id_") and node.id_:
            raw = node.id_
        else:
            raw = f"{default_prefix}_{node.element_id}"
        sid = re.sub(r"[^A-Za-z0-9_]", "_", raw)
        if not re.match(r"[A-Za-z_]", sid):
            sid = f"{default_prefix}_{sid}"
        return sid

    @staticmethod
    def get_notes(node):
        """
        Return the XHTML notes of a node, decoding `notes_base64` if needed.
        """
        if getattr(node, "notes", None):
            return node.notes
        if getattr(node, "notes_base64", None):
            return base64.b64decode(node.notes_base64).decode()
        return None

    @classmethod
    def set_notes(cls, node, sbml_element):
        notes = cls.get_notes(node)
        if notes and sbml_element.setNotes(notes) != libsbml.LIBSBML_OPERATION_SUCCESS:
            print(f"⚠ Could not set notes on '{sbml_element.getId()}'")


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
            self.set_notes(comp, c)


# ============================================================================
# Species Writer
# ============================================================================

class SBMLSpeciesWriter(SBMLWriter):

    DEFAULT_COMPARTMENT = "default_compartment"

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
            s.setCompartment(self._compartment_id(node))

            # Required in L3; the KG does not store these, so use SBML defaults
            s.setHasOnlySubstanceUnits(False)
            s.setBoundaryCondition(False)
            s.setConstant(False)

            self.set_notes(node, s)

    def _compartment_id(self, node):
        """
        Compartment from the "contained entity" edge; species without one
        are placed in a default compartment, since SBML L3 requires it.
        """
        compartments = node.compartment.all()
        if len(compartments) > 1:
            print(f"⚠ Species '{node.id_}' is in {len(compartments)} compartments, using the first")
        if compartments:
            return self.to_sid(compartments[0], "comp")

        if self.sbml_model.getCompartment(self.DEFAULT_COMPARTMENT) is None:
            c = self.sbml_model.createCompartment()
            c.setId(self.DEFAULT_COMPARTMENT)
            c.setConstant(True)
            c.setSBOTerm("SBO:0000290")
        return self.DEFAULT_COMPARTMENT


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
            # Required in L3; reversibility is not stored in the KG
            r.setReversible(False)
            self.set_notes(proc, r)

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
            sr.setConstant(True)
            if rel.stoichiometry is not None:
                sr.setStoichiometry(rel.stoichiometry)

    def _write_products(self, proc, sbml_reaction):
        products = proc.product
        for species in products:
            rel = proc.product.relationship(species)
            sp = sbml_reaction.createProduct()
            sp.setSpecies(self.to_sid(species, "species"))
            sp.setConstant(True)
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

        self.sbml_model.setId(SBMLWriter.to_sid(self.model, "model"))
        self.sbml_model.setName(self.model.name or self.sbml_model.getId())

        if self.model.sbo:
            self.sbml_model.setSBOTerm(self.model.sbo)
        SBMLWriter.set_notes(self.model, self.sbml_model)

    def _validate(self):
        """
        Run libSBML consistency checks and report errors.
        Returns the number of errors.

        Units and modeling-practice checks are skipped: the KG does not store
        units or initial values, so these would only produce noise.
        """
        self.doc.setConsistencyChecks(libsbml.LIBSBML_CAT_UNITS_CONSISTENCY, False)
        self.doc.setConsistencyChecks(libsbml.LIBSBML_CAT_MODELING_PRACTICE, False)
        self.doc.checkConsistency()
        n_errors = 0
        for i in range(self.doc.getNumErrors()):
            err = self.doc.getError(i)
            if err.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
                n_errors += 1
                print(f"  [{err.getSeverityAsString()}] {err.getMessage().strip()}")
        return n_errors

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

        n_errors = self._validate()
        if n_errors:
            print(f"⚠ {n_errors} SBML validation error(s), writing anyway")

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

    config = get_config()
    config.database_url = 'bolt://neo4j:password@localhost:7687'  # default

    args = parse_args()

    model_id = args.model_id
    outfile = args.outfile or f"{model_id}.xml"

    exporter = SBMLExporter(model_id=model_id)
    exporter.export(outfile)

if __name__ == "__main__":
    main()

