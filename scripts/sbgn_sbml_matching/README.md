# Scripts
## SBGN Gilda Annotator
* sbgn_gilda_annotator.py: Annotate a single or folder of SBGN files using Gilda (https://grounding.indra.bio/)

```
uv run sbgn_gilda_annotator.py -f sbgn -o sbgn_annotated
```

## SBGN-SBML Matching
* sbgn_sbml_identifiers_match.py: Pairwise comparison of annotated SBGN and SBML files; returns CSV of overlapping matches; by default assumes folders sbgn_annotated/ and sbml/ for content

```
uv run python sbgn_sbml_identifiers_match.py
```

# For Datasets (Google Drive)
## Annotation Format
Identifiers.org: http://identifiers.org/hgnc/6010

```
          <annotation>
            <rdf:RDF>
              <rdf:Description rdf:about="#Protein_85e81bba02cede0466dc486e9e380323_Complex_07a0860e7bac112d2c355e2f9f98f6f0">
                <bqmodel:is>
                  <rdf:Bag>
                    <rdf:li rdf:resource="http://identifiers.org/hgnc/6010" />
                    <rdf:li rdf:resource="http://identifiers.org/uniprot/P31785" />
                  </rdf:Bag>
                </bqmodel:is>
              </rdf:Description>
            </rdf:RDF>
          </annotation>
```

## sbgn_annotated.zip
* Source: Reactome subset (~1700): https://github.com/datapplab/SBGNhub/tree/master/data/SBGN/pathwayCommons with data coming from the Pathway Commons API that can generate SBGN https://www.pathwaycommons.org/
* Annotations: https://github.com/biocypher/sys-bio-kgs/tree/main/scripts
* ~1500 files were able to have 1 annotation

## sbml.zip
* Source: Biomodels 415 curated (should have annotations), SBML modeling human biology

## sbgn_sbml_identifier_overlap.csv.zip
Pairwise comparison of annotations for SBML and SBGN files with counts of overlapping unique identifiers
