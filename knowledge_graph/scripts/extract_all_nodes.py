import json
import re
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEANED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "nodes"
)

OUTPUT_FILE = OUTPUT_DIR / "all_nodes.json"

# Temporary output.
# The final all_nodes.json will only be replaced after
# successful processing of all judgments.
TEMP_OUTPUT_FILE = OUTPUT_DIR / "all_nodes_new.json"


# ============================================================
# LOAD CASE
# ============================================================

def load_case(case_file):
    """
    Load one cleaned judgment JSON file.
    """

    with open(
        case_file,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# GET FULL JUDGMENT TEXT
# ============================================================

def get_full_text(case_data):
    """
    Combine cleaned text from all pages.
    """

    return "\n".join(
        page.get("cleaned_text", "")
        for page in case_data.get("pages", [])
        if page.get("cleaned_text")
    )


# ============================================================
# CASE METADATA
# ============================================================

def extract_basic_metadata(
    case_data,
    full_text,
    case_file
):
    """
    Extract basic case metadata.

    source_file is obtained safely from:
        1. case_data["source_file"]
        2. actual cleaned JSON filename
    """

    case_id = case_data.get(
        "case_id",
        case_file.stem
    )

    metadata = {
        "case_id": case_id,
        "case_name": None,
        "court": case_data.get("court"),
        "date": None,
        "judges": [],
        "source_file": case_data.get(
            "source_file",
            case_file.name
        )
    }

    # --------------------------------------------------------
    # Case name
    # --------------------------------------------------------

    if "_on_" in case_id:

        metadata["case_name"] = (
            case_id
            .rsplit("_on_", 1)[0]
            .replace("_", " ")
        )

    else:

        metadata["case_name"] = (
            case_id.replace("_", " ")
        )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    date_match = re.search(
        r"_on_(\d{1,2})_([A-Za-z]+)_(\d{4})$",
        case_id
    )

    if date_match:

        metadata["date"] = (
            f"{date_match.group(1)} "
            f"{date_match.group(2)} "
            f"{date_match.group(3)}"
        )

    # --------------------------------------------------------
    # Judges / Bench
    # --------------------------------------------------------

    bench_match = re.search(
        r"Bench:\s*([^\n]+)",
        full_text,
        re.IGNORECASE
    )

    if bench_match:

        metadata["judges"] = [
            name.strip()
            for name in bench_match.group(1).split(",")
            if name.strip()
        ]

    return metadata


# ============================================================
# CONSTITUTIONAL ARTICLE EXTRACTION
# ============================================================

def extract_articles(full_text):
    """
    Extract constitutional Articles.

    Examples:
        Article 14
        Article 21
        Article 31C
        Article 226
    """

    found = set()

    normalized_text = re.sub(
        r"\s+",
        " ",
        full_text
    )

    pattern = r"\bArticle\s+(\d+[A-Za-z]?)\b"

    for match in re.finditer(
        pattern,
        normalized_text,
        re.IGNORECASE
    ):

        article_number = match.group(1)

        found.add(
            f"Article {article_number}"
        )

    return sorted(
        found,
        key=lambda x: (
            int(re.search(r"\d+", x).group()),
            x
        )
    )


# ============================================================
# STATUTORY SECTION EXTRACTION
# ============================================================

def extract_statutory_sections(full_text):
    """
    Extract statutory section references.

    Examples:
        Section 438
        Section 302
        Section 420
        Section 498A

    Also handles:
        Sec 438
        Sec. 438

    IMPORTANT:
    We extract the section number only.

    We do NOT infer the Act automatically.
    """

    found = set()

    normalized_text = re.sub(
        r"\s+",
        " ",
        full_text
    )

    pattern = (
        r"\b(?:Section|Sec\.?)\s+"
        r"(\d+[A-Za-z]?)\b"
    )

    for match in re.finditer(
        pattern,
        normalized_text,
        re.IGNORECASE
    ):

        section_number = match.group(1)

        found.add(
            f"Section {section_number}"
        )

    return sorted(
        found,
        key=lambda x: (
            int(re.search(r"\d+", x).group()),
            x
        )
    )


# ============================================================
# CASE REFERENCE EXTRACTION
# ============================================================

def extract_case_references(full_text):
    """
    Extract references to other cases.

    The extractor is intentionally conservative.

    It also limits the amount of text processed for this
    expensive regex operation so that one unusually large
    judgment cannot stall the entire KG build.
    """

    # --------------------------------------------------------
    # Normalize whitespace
    # --------------------------------------------------------

    normalized_text = re.sub(
        r"\s+",
        " ",
        full_text
    )

    # --------------------------------------------------------
    # Safety limit
    #
    # We only need enough judgment text to identify case
    # references. The original KG already demonstrated that
    # citation extraction can become noisy.
    # --------------------------------------------------------

    MAX_REFERENCE_TEXT = 500000

    if len(normalized_text) > MAX_REFERENCE_TEXT:

        normalized_text = (
            normalized_text[:MAX_REFERENCE_TEXT]
        )

    references = []

    # --------------------------------------------------------
    # Case-name pattern
    # --------------------------------------------------------

    pattern = re.compile(
        r"\b"
        r"([A-Z][A-Za-z.'’&()\-\*]*"
        r"(?:\s+[A-Z][A-Za-z.'’&()\-\*]*){0,6})"
        r"\s+(?:v\.|vs\.|versus)\s+"
        r"([A-Z][A-Za-z.'’&()\-\*]*"
        r"(?:\s+[A-Za-z][A-Za-z.'’&()\-\*]*){0,6})",
        re.IGNORECASE
    )

    for match in pattern.finditer(
        normalized_text
    ):

        petitioner = " ".join(
            match.group(1).split()
        )

        respondent = " ".join(
            match.group(2).split()
        )

        # ----------------------------------------------------
        # Remove common prose prefixes
        # ----------------------------------------------------

        petitioner = re.sub(
            r"^(?:in|i|ii|iii|iv|v|vi|"
            r"and|that|where|from|by|under|"
            r"notices that|held in|observed in)\s+",
            "",
            petitioner,
            flags=re.IGNORECASE
        ).strip()

        # ----------------------------------------------------
        # Remove common trailing prose
        # ----------------------------------------------------

        respondent = re.sub(
            r"\s+(?:and|but|which|where|"
            r"the|this|that|has|had|was|were)$",
            "",
            respondent,
            flags=re.IGNORECASE
        ).strip()

        # ----------------------------------------------------
        # Basic length checks
        # ----------------------------------------------------

        if len(petitioner) < 4:
            continue

        if len(respondent) < 4:
            continue

        # ----------------------------------------------------
        # Reject obvious prose prefixes
        # ----------------------------------------------------

        prose_prefixes = [
            "the supreme court in",
            "the high court in",
            "the madras high court in",
            "the delhi high court in",
            "the court in",
            "through",
            "in",
            "and",
        ]

        petitioner_lower = petitioner.lower()

        if any(
            petitioner_lower.startswith(
                prefix + " "
            )
            for prefix in prose_prefixes
        ):
            continue

        # ----------------------------------------------------
        # Reject incomplete respondents
        # ----------------------------------------------------

        if respondent.lower().strip() in [
            "registrar general of the high court of",
            "high court of",
            "union of",
            "state of",
        ]:
            continue

        # ----------------------------------------------------
        # Require meaningful words
        # ----------------------------------------------------

        if not any(
            len(word) >= 4
            for word in petitioner.split()
        ):
            continue

        if not any(
            len(word) >= 4
            for word in respondent.split()
        ):
            continue

        # ----------------------------------------------------
        # Reject known prose patterns
        # ----------------------------------------------------

        rejected_phrases = [
            "inclusion",
            "excellence",
            "decision",
            "judgment",
            "reference",
            "petition",
            "petitioners",
            "respondents",
            "this court",
            "the court",
            "the following",
            "the present",
            "under the",
            "for the",
            "with the",
            "from the",
            "before the",
            "against the",
        ]

        candidate_lower = (
            f"{petitioner} v. {respondent}"
        ).lower()

        if any(
            phrase in candidate_lower
            for phrase in rejected_phrases
        ):
            continue

        # ----------------------------------------------------
        # Reject obvious prose fragments
        # ----------------------------------------------------

        bad_words = [
            "petitioners",
            "respondents",
            "judgment",
            "paragraph",
            "decision",
            "bench",
            "candidates",
            "context",
            "principle",
            "rubric",
            "assistance",
            "appeared",
            "issued",
            "writ petition",
            "present case",
        ]

        if any(
            word in candidate_lower.split()
            for word in bad_words
        ):
            continue

        # ----------------------------------------------------
        # Create candidate
        # ----------------------------------------------------

        candidate = (
            f"{petitioner} v. {respondent}"
        )

        if len(candidate) > 120:
            continue

        # ----------------------------------------------------
        # Remove citation-report suffixes
        # ----------------------------------------------------

        candidate = re.sub(
            r"\s+(?:AIR|SCC|OnLine|SCR)\s*$",
            "",
            candidate,
            flags=re.IGNORECASE
        ).strip()

        if candidate not in references:

            references.append(
                candidate
            )

    # ========================================================
    # REMOVE PARTIAL DUPLICATES
    # ========================================================

    cleaned_references = []

    for ref in references:

        normalized_ref = re.sub(
            r"[^a-z0-9]+",
            " ",
            ref.lower()
        ).strip()

        is_duplicate = False

        for existing in cleaned_references:

            existing_norm = re.sub(
                r"[^a-z0-9]+",
                " ",
                existing.lower()
            ).strip()

            if normalized_ref in existing_norm:

                is_duplicate = True
                break

        if not is_duplicate:

            cleaned_references.append(
                ref
            )

    return cleaned_references[:50]


# ============================================================
# BUILD NODES FOR ONE CASE
# ============================================================

def build_case_nodes(
    case_data,
    case_file
):

    # --------------------------------------------------------
    # Full text
    # --------------------------------------------------------

    full_text = get_full_text(
        case_data
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = extract_basic_metadata(
        case_data,
        full_text,
        case_file
    )

    # --------------------------------------------------------
    # Constitutional Articles
    # --------------------------------------------------------

    articles = extract_articles(
        full_text
    )

    # --------------------------------------------------------
    # Statutory Sections
    # --------------------------------------------------------

    statutory_sections = (
        extract_statutory_sections(
            full_text
        )
    )

    # --------------------------------------------------------
    # Case references
    # --------------------------------------------------------

    case_references = (
        extract_case_references(
            full_text
        )
    )

    case_id = metadata["case_id"]

    nodes = []
    relationships = []

    # ========================================================
    # CASE NODE
    # ========================================================

    nodes.append({

        "node_type": "Case",

        "node_id": case_id,

        "properties": {

            "case_id": case_id,

            "case_name": metadata["case_name"],

            "court": metadata["court"],

            "date": metadata["date"],

            "source_file": metadata.get(
                "source_file",
                case_file.name
            )
        }
    })

    # ========================================================
    # COURT NODE
    # ========================================================

    court = metadata["court"]

    if court:

        court_id = (
            court
            .strip()
            .replace(" ", "_")
            .lower()
        )

        nodes.append({

            "node_type": "Court",

            "node_id": court_id,

            "properties": {

                "court_name": court
            }
        })

        relationships.append({

            "relationship_type": "BELONGS_TO",

            "source": case_id,

            "target": court_id
        })

    # ========================================================
    # JUDGE NODES
    # ========================================================

    for judge in metadata["judges"]:

        judge_id = (
            "judge_"
            + re.sub(
                r"[^a-z0-9]+",
                "_",
                judge.lower()
            ).strip("_")
        )

        nodes.append({

            "node_type": "Judge",

            "node_id": judge_id,

            "properties": {

                "judge_name": judge
            }
        })

        relationships.append({

            "relationship_type": "DECIDED_BY",

            "source": case_id,

            "target": judge_id
        })

    # ========================================================
    # CONSTITUTIONAL ARTICLE NODES
    # ========================================================

    for article in articles:

        article_id = (
            "section_"
            + re.sub(
                r"[^a-z0-9]+",
                "_",
                article.lower()
            ).strip("_")
        )

        nodes.append({

            "node_type": "Section",

            "node_id": article_id,

            "properties": {

                "section_name": article,

                "section_type": (
                    "constitutional_article"
                )
            }
        })

        relationships.append({

            "relationship_type": "INVOLVES_SECTION",

            "source": case_id,

            "target": article_id
        })

    # ========================================================
    # STATUTORY SECTION NODES
    # ========================================================

    for section in statutory_sections:

        section_id = (
            "section_"
            + re.sub(
                r"[^a-z0-9]+",
                "_",
                section.lower()
            ).strip("_")
        )

        nodes.append({

            "node_type": "Section",

            "node_id": section_id,

            "properties": {

                "section_name": section,

                "section_type": "statutory"
            }
        })

        relationships.append({

            "relationship_type": "INVOLVES_SECTION",

            "source": case_id,

            "target": section_id
        })

    # ========================================================
    # CITED CASE NODES
    # ========================================================

    for reference in case_references:

        cited_case_id = (
            "case_ref_"
            + re.sub(
                r"[^a-z0-9]+",
                "_",
                reference.lower()
            ).strip("_")
        )

        nodes.append({

            "node_type": "Case",

            "node_id": cited_case_id,

            "properties": {

                "case_name": reference,

                "reference_only": True
            }
        })

        relationships.append({

            "relationship_type": "CITES",

            "source": case_id,

            "target": cited_case_id
        })

    return nodes, relationships


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    case_files = sorted(
        CLEANED_DIR.glob("*.json")
    )

    print(
        "\n=========================================="
    )

    print(
        "INDIAN LEGAL KNOWLEDGE GRAPH EXTRACTION"
    )

    print(
        "==========================================\n"
    )

    print(
        f"Cleaned judgment files found : "
        f"{len(case_files)}"
    )

    print()

    all_nodes = []
    all_relationships = []

    successful_cases = 0
    failed_cases = 0

    # ========================================================
    # PROCESS ALL JUDGMENTS
    # ========================================================

    for index, case_file in enumerate(
        case_files,
        start=1
    ):

        try:

            case_data = load_case(
                case_file
            )

            nodes, relationships = (
                build_case_nodes(
                    case_data,
                    case_file
                )
            )

            all_nodes.extend(
                nodes
            )

            all_relationships.extend(
                relationships
            )

            successful_cases += 1

            print(
                f"[{index:03d}/{len(case_files):03d}] "
                f"Processed: {case_file.name}"
            )

        except Exception as error:

            failed_cases += 1

            print(
                f"[ERROR] {case_file.name}"
            )

            print(
                f"        {error}"
            )

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if successful_cases == 0:

        print(
            "\nERROR: No judgments were successfully "
            "processed."
        )

        print(
            "The existing knowledge graph was NOT replaced."
        )

        return

    if failed_cases > 0:

        print(
            f"\nWARNING: {failed_cases} judgment(s) "
            f"failed during extraction."
        )

        print(
            "The existing knowledge graph will NOT "
            "be replaced."
        )

        print(
            "Fix the extraction errors before rebuilding "
            "the final KG."
        )

        return

    # ========================================================
    # REMOVE DUPLICATE NODES
    # ========================================================

    unique_nodes = {}

    for node in all_nodes:

        key = (
            node["node_type"],
            node["node_id"]
        )

        unique_nodes[key] = node

    all_nodes = list(
        unique_nodes.values()
    )

    # ========================================================
    # REMOVE DUPLICATE RELATIONSHIPS
    # ========================================================

    unique_relationships = {}

    for relationship in all_relationships:

        key = (
            relationship["relationship_type"],
            relationship["source"],
            relationship["target"]
        )

        unique_relationships[key] = relationship

    all_relationships = list(
        unique_relationships.values()
    )

    # ========================================================
    # FINAL GRAPH DATA
    # ========================================================

    graph_data = {

        "metadata": {

            "total_judgments_found": len(
                case_files
            ),

            "successful_judgments": (
                successful_cases
            ),

            "failed_judgments": (
                failed_cases
            ),

            "total_unique_nodes": (
                len(all_nodes)
            ),

            "total_unique_relationships": (
                len(all_relationships)
            )
        },

        "nodes": all_nodes,

        "relationships": all_relationships
    }

    # ========================================================
    # SAVE TEMPORARY GRAPH
    # ========================================================

    with open(
        TEMP_OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            graph_data,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # NODE TYPE COUNTS
    # ========================================================

    node_counts = {}

    for node in all_nodes:

        node_type = node["node_type"]

        node_counts[node_type] = (
            node_counts.get(
                node_type,
                0
            ) + 1
        )

    # ========================================================
    # RELATIONSHIP COUNTS
    # ========================================================

    relationship_counts = {}

    for relationship in all_relationships:

        relationship_type = (
            relationship["relationship_type"]
        )

        relationship_counts[
            relationship_type
        ] = (
            relationship_counts.get(
                relationship_type,
                0
            ) + 1
        )

    # ========================================================
    # SECTION TYPE COUNTS
    # ========================================================

    constitutional_article_count = 0
    statutory_section_count = 0

    for node in all_nodes:

        if node["node_type"] != "Section":
            continue

        section_type = (
            node
            .get("properties", {})
            .get("section_type")
        )

        if (
            section_type
            == "constitutional_article"
        ):

            constitutional_article_count += 1

        elif section_type == "statutory":

            statutory_section_count += 1

    # ========================================================
    # FINAL REPLACEMENT
    # ========================================================

    TEMP_OUTPUT_FILE.replace(
        OUTPUT_FILE
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "EXTRACTION COMPLETE"
    )

    print(
        "==========================================\n"
    )

    print(
        f"Judgments found     : "
        f"{len(case_files)}"
    )

    print(
        f"Successfully parsed : "
        f"{successful_cases}"
    )

    print(
        f"Failed              : "
        f"{failed_cases}"
    )

    print(
        f"Unique nodes        : "
        f"{len(all_nodes)}"
    )

    print(
        f"Relationships       : "
        f"{len(all_relationships)}"
    )

    print(
        "\nSECTION BREAKDOWN"
    )

    print(
        "------------------"
    )

    print(
        f"Constitutional Articles : "
        f"{constitutional_article_count}"
    )

    print(
        f"Statutory Sections      : "
        f"{statutory_section_count}"
    )

    print(
        "\nNODE COUNTS"
    )

    print(
        "----------------"
    )

    for node_type, count in sorted(
        node_counts.items()
    ):

        print(
            f"{node_type:15s}: {count}"
        )

    print(
        "\nRELATIONSHIP COUNTS"
    )

    print(
        "----------------"
    )

    for relationship_type, count in sorted(
        relationship_counts.items()
    ):

        print(
            f"{relationship_type:20s}: {count}"
        )

    print(
        "\nOUTPUT FILE"
    )

    print(
        "----------------"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\n==========================================\n"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()