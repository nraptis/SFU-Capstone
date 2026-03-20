from dataclasses import dataclass

@dataclass(frozen=True)
class CellTypeCard:
    title: str
    tagline: str
    summary: str
    concern: str

CELL_TYPE_CARDS = [
    CellTypeCard(
        title="Plasma Cell",
        tagline='“Antibody Factory.”',
        summary="Rare in normal peripheral blood; usually lives in marrow or lymph nodes.",
        concern="High concern: uncommon in blood. Often linked to plasma cell disorders; sometimes severe immune activation.",
    ),
    CellTypeCard(
        title="Myeloblast",
        tagline='“Stem-Stage Precursor.”',
        summary="Very immature myeloid cell. Seeing blasts in blood is a major abnormal finding.",
        concern="High concern: can be associated with acute leukemia or marrow failure syndromes; needs clinical correlation.",
    ),
    CellTypeCard(
        title="Promyelocyte",
        tagline='“Young Granulocyte.”',
        summary="Very early granulocyte precursor. Presence in blood is highly unusual.",
        concern="Critical concern: classically linked with APL risk patterns; urgent clinical follow-up is standard.",
    ),
    CellTypeCard(
        title="Promyelocyte (Atypical)",
        tagline='“Auer Rod Bundles.”',
        summary="Abnormal promyelocyte morphology; may show dense granules and Auer rod bundles.",
        concern="Very high concern: strongly associated with APL-type presentations in clinical settings.",
    ),
    CellTypeCard(
        title="Myelocyte",
        tagline='“Middle-Stage Precursor.”',
        summary="Often the earliest immature granulocyte seen in a benign left shift.",
        concern="Mixed concern: may appear with severe infection/inflammation; persistent elevation can suggest marrow disorders.",
    ),
    CellTypeCard(
        title="Metamyelocyte",
        tagline='“Late-Stage Teen.”',
        summary="Immature granulocyte with kidney-bean nucleus; part of the left-shift spectrum.",
        concern="Inflammatory pattern: can rise in severe infection/stress; context and counts matter.",
    ),
    CellTypeCard(
        title="Neutrophil Band",
        tagline='“First Left-Shift Step.”',
        summary="Slightly immature but functional neutrophil; common in acute bacterial responses.",
        concern="Often reactive: frequently seen with bacterial infection or physiologic stress; low cancer specificity alone.",
    ),
    CellTypeCard(
        title="Neutrophil Segmented",
        tagline='“Adult Soldier.”',
        summary="Most common WBC. Primary defense against bacteria and acute inflammation.",
        concern="Baseline finding: high counts often infection/steroids; low counts can occur with meds, autoimmune, or sepsis.",
    ),
    CellTypeCard(
        title="Monocyte",
        tagline='“Garbage Collector.”',
        summary="Large phagocyte that clears debris and helps coordinate immune response.",
        concern="Reactive pattern: may rise in chronic infection/inflammation or recovery states; persistent monocytosis needs evaluation.",
    ),
    CellTypeCard(
        title="Lymphocyte",
        tagline='“Special Ops.”',
        summary="Includes T- and B-cells; commonly elevated with viral immune responses.",
        concern="Often reactive: viral illness can raise counts; persistent high counts may require hematology workup depending on context.",
    ),
    CellTypeCard(
        title="Lymphocyte (Large Granular)",
        tagline='“Killer Subtype.”',
        summary="Often NK/T lineage with prominent granules; participates in cytotoxic defense.",
        concern="Mixed: can appear with viral/reactive states; rarely linked to LGL-type disorders when persistent.",
    ),
    CellTypeCard(
        title="Lymphocyte (Neoplastic)",
        tagline='“Clonal Pattern.”',
        summary="Atypical lymphocyte features can suggest a clonal process in the right clinical context.",
        concern="High concern: may align with leukemia/lymphoma patterns; requires confirmatory clinical testing.",
    ),
    CellTypeCard(
        title="Hairy Cell",
        tagline='“Frayed Edges.”',
        summary="Lymphoid cell with characteristic ‘hair-like’ projections; morphology can be distinctive.",
        concern="High concern when confirmed: classically associated with hairy cell leukemia; diagnosis is clinical and lab-confirmed.",
    ),
    CellTypeCard(
        title="Basophil",
        tagline='“Allergy Alarm.”',
        summary="Rare granulocyte involved in histamine signaling; typically low in normal blood.",
        concern="Specific clue: can rise with allergic states; basophilia can also appear in myeloproliferative disorders.",
    ),
    CellTypeCard(
        title="Eosinophil",
        tagline='“Parasite Hunter.”',
        summary="Granulocyte active in allergy/asthma pathways and some parasitic responses.",
        concern="Often reactive: allergies/asthma/parasites; persistent marked elevation can require further evaluation.",
    ),
    CellTypeCard(
        title="Normoblast",
        tagline='“Nucleated Red Cell.”',
        summary="An immature red blood cell that still contains a nucleus. Normally confined to bone marrow, not peripheral blood.",
        concern="High concern in adults: may be seen with severe anemia, marrow stress, hypoxia, or marrow infiltration. Clinical context is essential.",
    ),
]