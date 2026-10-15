import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned"
    / "Abhimeet_Sinha_vs_High_Court_Of_Judicature_At_Patna_on_6_May_2024.json"
)


def load_case():
    with open(CASE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_full_text(case_data):
    return "\n".join(
        page["cleaned_text"]
        for page in case_data["pages"]
        if page.get("cleaned_text")
    )


def extract_basic_metadata(case_data, full_text):
    metadata = {
        "case_id": case_data.get("case_id"),
        "court": case_data.get("court"),
        "source_file": case_data.get("source_file"),
        "case_name": None,
        "date": None,
        "judges": [],
    }

    # Case name from the case_id
    case_id = case_data.get("case_id", "")
    metadata["case_name"] = (
        case_id.rsplit("_on_", 1)[0].replace("_", " ")
    )

    # Date from case_id
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

    # Bench line
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


def extract_articles(full_text):
    found = set()

    # Normalize whitespace
    normalized_text = re.sub(r"\s+", " ", full_text)

    # Single article references
    single_pattern = r"\bArticle\s+(\d+[A-Za-z]?)\b"

    for match in re.finditer(
        single_pattern,
        normalized_text,
        re.IGNORECASE
    ):
        article_number = match.group(1)
        found.add(f"Article {article_number}")

    return sorted(
        found,
        key=lambda x: (
            int(re.search(r"\d+", x).group()),
            x
        )
    )


def extract_case_references(full_text):
    normalized_text = re.sub(r"\s+", " ", full_text)

    references = []

    # ---------------------------------------------------------
    # CURRENT CASE
    # ---------------------------------------------------------

    current_case_name = (
        "Abhimeet Sinha vs High Court Of Judicature At Patna"
    )

    current_case_normalized = re.sub(
        r"[^a-z0-9]+",
        " ",
        current_case_name.lower()
    ).strip()

    # ---------------------------------------------------------
    # CASE CITATION PATTERN
    # ---------------------------------------------------------
    #
    # IMPORTANT:
    # No IGNORECASE flag here.
    #
    # This makes the first word of each party name normally
    # begin with an uppercase character, which helps prevent
    # ordinary prose from becoming a case citation.
    # ---------------------------------------------------------

    pattern = re.compile(
        r"\b"
        r"([A-Z][A-Za-z0-9.'’&()\-]*"
        r"(?:\s+[A-Za-z0-9.'’&()\-]+){0,7})"
        r"\s+(?:v\.?|vs\.?|versus)\s+"
        r"([A-Z][A-Za-z0-9.'’&()\-]*"
        r"(?:\s+[A-Za-z0-9.'’&()\-]+){0,7})"
    )

    for match in pattern.finditer(normalized_text):

        petitioner = " ".join(
            match.group(1).split()
        ).strip()

        respondent = " ".join(
            match.group(2).split()
        ).strip()

        # -----------------------------------------------------
        # REMOVE LEADING NUMBERS
        # -----------------------------------------------------

        petitioner = re.sub(
            r"^\d+\s+",
            "",
            petitioner
        ).strip()

        respondent = re.sub(
            r"^\d+\s+",
            "",
            respondent
        ).strip()

        # -----------------------------------------------------
        # REMOVE REPORTING INFORMATION
        #
        # Examples:
        #
        # (1995) 3 SCC 486
        # 2000 3 SCC 588
        # 2017(9) SCC 1
        # (2015) 11 SCC 493
        # -----------------------------------------------------

        respondent = re.sub(
            r"\s*\(\s*\d{4}\s*\).*",
            "",
            respondent
        ).strip()

        respondent = re.sub(
            r"\s+\d{4}\s*\(\s*\d+\s*\).*",
            "",
            respondent
        ).strip()

        respondent = re.sub(
            r"\s+\d{4}\s+\d+\s+"
            r"(?:SCC|SCR|AIR|SCC\s+OnLine|OnLine)"
            r".*",
            "",
            respondent,
            flags=re.IGNORECASE
        ).strip()

        respondent = re.sub(
            r"\s+(?:AIR|SCC|SCR|OnLine)\b.*",
            "",
            respondent,
            flags=re.IGNORECASE
        ).strip()

        # -----------------------------------------------------
        # REMOVE COMMON TRAILING PROSE
        # -----------------------------------------------------

        respondent = re.sub(
            r"\s+(?:"
            r"for short.*|"
            r"dated.*|"
            r"and others dated.*|"
            r"etc\..*"
            r")$",
            "",
            respondent,
            flags=re.IGNORECASE
        ).strip()

        # -----------------------------------------------------
        # REMOVE OBVIOUS LEADING PROSE FROM PETITIONER
        # -----------------------------------------------------

        petitioner_lower = petitioner.lower()

        prose_prefixes = [
            "court in",
            "the court in",
            "supreme court in",
            "the supreme court in",
            "high court in",
            "the high court in",
            "case of",
            "the case of",
            "matter of",
            "the matter of",
            "question of",
            "the question of",
            "issue of",
            "the issue of",
            "absence of",
            "the absence of",
            "presence of",
            "the presence of",
            "purpose of",
            "the purpose of",
            "process of",
            "the process of",
            "decision in",
            "the decision in",
            "judgment in",
            "the judgment in"
        ]

        if any(
            petitioner_lower.startswith(prefix)
            for prefix in prose_prefixes
        ):
            continue

        # -----------------------------------------------------
        # BASIC LENGTH CHECK
        # -----------------------------------------------------

        if len(petitioner) < 4:
            continue

        if len(respondent) < 4:
            continue

        # -----------------------------------------------------
        # REJECT CLEAR EXTRACTION NOISE
        # -----------------------------------------------------

        bad_fragments = [
            "advocate)",
            "applicant)",
            "petitioner)",
            "respondent)",
            "paragraph",
            "judgment",
            "decision",
            "bench",
            "candidates",
            "context",
            "rubric",
            "assistance",
            "appeared",
            "issued",
            "writ petition",
            "present case"
        ]

        candidate_lower = (
            f"{petitioner} v. {respondent}"
        ).lower()

        if any(
            fragment in candidate_lower
            for fragment in bad_fragments
        ):
            continue

        # -----------------------------------------------------
        # REMOVE TRAILING CONNECTORS
        # -----------------------------------------------------

        respondent = re.sub(
            r"\s+(?:and|but|which|where|the|this|that|has|had|was|were)$",
            "",
            respondent,
            flags=re.IGNORECASE
        ).strip()

        # -----------------------------------------------------
        # CHECK MEANINGFUL WORDS
        # -----------------------------------------------------

        if not any(
            len(re.sub(r"[^A-Za-z]", "", word)) >= 4
            for word in petitioner.split()
        ):
            continue

        if not any(
            len(re.sub(r"[^A-Za-z]", "", word)) >= 4
            for word in respondent.split()
        ):
            continue

        # -----------------------------------------------------
        # REMOVE NUMBERED PARAGRAPH MARKERS
        # -----------------------------------------------------

        petitioner = re.sub(
            r"^\d+\.\s*",
            "",
            petitioner
        ).strip()

        respondent = re.sub(
            r"^\d+\.\s*",
            "",
            respondent
        ).strip()

        # -----------------------------------------------------
        # BUILD CANDIDATE
        # -----------------------------------------------------

        candidate = (
            f"{petitioner} v. {respondent}"
        )

        # -----------------------------------------------------
        # LIMIT SIZE
        # -----------------------------------------------------

        if len(candidate) > 120:
            continue

        if len(petitioner.split()) > 9:
            continue

        if len(respondent.split()) > 9:
            continue

        # -----------------------------------------------------
        # EXCLUDE CURRENT CASE
        # -----------------------------------------------------

        candidate_normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            candidate.lower()
        ).strip()

        if (
            candidate_normalized
            == current_case_normalized
        ):
            continue

        # -----------------------------------------------------
        # UNIQUE REFERENCES
        # -----------------------------------------------------

        if candidate not in references:

            references.append(candidate)

    # ---------------------------------------------------------
    # REMOVE EXACT DUPLICATES
    # ---------------------------------------------------------

    cleaned_references = []

    seen = set()

    for reference in references:

        normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            reference.lower()
        ).strip()

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)

        cleaned_references.append(
            reference
        )

    return cleaned_references[:50]
def main():

    case_data = load_case()
    full_text = get_full_text(case_data)

    metadata = extract_basic_metadata(
        case_data,
        full_text
    )

    articles = extract_articles(full_text)

    case_references = extract_case_references(
        full_text
    )

    print("\n=== KNOWLEDGE GRAPH EXTRACTION TEST ===\n")

    print("CASE METADATA")
    print("----------------")

    print(f"Case ID : {metadata['case_id']}")
    print(f"Case    : {metadata['case_name']}")
    print(f"Court   : {metadata['court']}")
    print(f"Date    : {metadata['date']}")
    print(f"Judges  : {metadata['judges']}")

    print("\nARTICLES FOUND")
    print("----------------")

    for article in articles[:20]:
        print(article)

    print("\nCASE REFERENCES FOUND")
    print("----------------")

    for reference in case_references[:30]:
        print(reference)

    print("\n=== END TEST ===")


if __name__ == "__main__":
    main()