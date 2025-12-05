from neomodel import (
    StructuredNode,
    StructuredRel,
    StringProperty,
    FloatProperty,
    ArrayProperty,
    RelationshipTo,
    RelationshipFrom,
)

from neomodel import (
    StructuredNode,
    StructuredRel,
    StringProperty,
    FloatProperty,
    ArrayProperty,
)
# ---- Shared Properties for Nodes and Relationships ----
class QualifierMixin:
    """
    Shared SBO/momapy qualifier properties across all classes.
    """

    id_ = StringProperty(db_property="id")

    name = StringProperty()
    sbo = StringProperty()
    notes_base64 = StringProperty()
    notes = StringProperty()

    encodes = ArrayProperty(StringProperty())
    hasPart = ArrayProperty(StringProperty())
    hasProperty = ArrayProperty(StringProperty())
    hasVersion = ArrayProperty(StringProperty())
    is_ = ArrayProperty(StringProperty())
    isDescribedBy = ArrayProperty(StringProperty())
    isEncodedBy = ArrayProperty(StringProperty())
    isHomologTo = ArrayProperty(StringProperty())
    isPartOf = ArrayProperty(StringProperty())
    isPropertyOf = ArrayProperty(StringProperty())
    isVersionOf = ArrayProperty(StringProperty())
    occursIn = ArrayProperty(StringProperty())
    hasTaxon = ArrayProperty(StringProperty())
    hasInstance = ArrayProperty(StringProperty())
    isDerivedFrom = ArrayProperty(StringProperty())
    isInstanceOf = ArrayProperty(StringProperty())

# ---- ROOT: Systems Biology Representation ----
class SystemsBiologyRepresentation(StructuredNode, QualifierMixin):
    pass

# ---- Occurring Entity Representation ----
class OccurringEntityRepresentation(SystemsBiologyRepresentation):
    pass

# ---- Relationship (base for control/similarity/etc.) ----
class Relationship(OccurringEntityRepresentation):
    pass

# ---- Control (subclass of Relationship) ----
class Control(Relationship):
    pass


class Inhibition(Control):
    pass


class Stimulation(Control):
    pass


class Catalysis(Stimulation):
    pass


class NecessaryStimulation(Stimulation):
    pass

# ---- Similarity ----
class Similarity(Relationship):
    pass


# ---- Physical Entity Representation ----
class PhysicalEntityRepresentation(SystemsBiologyRepresentation):
    pass

# ---- Material Entity ----
class MaterialEntity(PhysicalEntityRepresentation):
    pass


class EmptySet(MaterialEntity):
    pass


class Macromolecule(MaterialEntity):
    pass


class InformationMacromolecule(Macromolecule):
    pass


class PhysicalCompartment(MaterialEntity):
    pass


# Relationship base class (for all SBO edges with shared properties)
class CommonRelProperties(StructuredRel):
    id_ = StringProperty(db_property="id")
    name = StringProperty()
    sbo = StringProperty()
    notes_base64 = StringProperty()
    notes = StringProperty()
    encodes = ArrayProperty(StringProperty())
    hasPart = ArrayProperty(StringProperty())
    hasProperty = ArrayProperty(StringProperty())
    hasVersion = ArrayProperty(StringProperty())
    is_ = ArrayProperty(StringProperty())
    isDescribedBy = ArrayProperty(StringProperty())
    isEncodedBy = ArrayProperty(StringProperty())
    isHomologTo = ArrayProperty(StringProperty())
    isPartOf = ArrayProperty(StringProperty())
    isPropertyOf = ArrayProperty(StringProperty())
    isVersionOf = ArrayProperty(StringProperty())
    occursIn = ArrayProperty(StringProperty())
    hasTaxon = ArrayProperty(StringProperty())
    hasInstance = ArrayProperty(StringProperty())
    isDerivedFrom = ArrayProperty(StringProperty())
    isInstanceOf = ArrayProperty(StringProperty())

# reactant (edge)
class ReactantRel(CommonRelProperties):
    stoichiometry = FloatProperty()

# product (edge)
class ProductRel(CommonRelProperties):
    stoichiometry = FloatProperty()

# modifier (edge)
class ModifierRel(CommonRelProperties):
    pass

# contained entity (edge)
class ContainedEntityRel(CommonRelProperties):
    pass

# Additional SBO edges
class InhibitionRel(CommonRelProperties):
    pass

class NecessaryStimulationRel(CommonRelProperties):
    pass

class CatalysisRel(CommonRelProperties):
    pass

class SimilarityRel(CommonRelProperties):
    pass

# ---- Process ----
class Process(OccurringEntityRepresentation):
    reactant = RelationshipFrom(PhysicalEntityRepresentation, "Reactant", model=ReactantRel)
    product = RelationshipTo(PhysicalEntityRepresentation, "Product", model=ProductRel)
    modifier = RelationshipFrom(PhysicalEntityRepresentation, "Modifier", model=ModifierRel)

# ---- Model ----
class Model(OccurringEntityRepresentation):
    compartment_of = RelationshipFrom(PhysicalCompartment, "IsCompartmentOf")
    entity_of = RelationshipFrom(PhysicalEntityRepresentation, "IsEntityOf")
    process_of = RelationshipFrom(Process, "IsProcessOf")

