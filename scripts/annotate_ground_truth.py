import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

CANDIDATES_FILE = Path(
    "data/evaluation/retrieval_candidates.json"
)

OUTPUT_FILE = Path(
    "data/evaluation/retrieval_eval_annotated.json"
)

CLEANED_DIR = Path(
    "data/processed/cleaned"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def save_json(data, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


def find_cleaned_case(case_id):

    exact_path = CLEANED_DIR / f"{case_id}.json"

    if exact_path.exists():
        return exact_path

    matches = list(
        CLEANED_DIR.glob(
            f"{case_id}*.json"
        )
    )

    if matches:
        return matches[0]

    return None


def display_case_page(case_id, page_number):

    case_path = find_cleaned_case(
        case_id
    )

    if case_path is None:

        print(
            "\n[WARNING] Cleaned case file "
            "could not be found."
        )

        return


    try:

        data = load_json(
            case_path
        )

    except Exception as error:

        print(
            f"\n[ERROR] Could not read case: "
            f"{error}"
        )

        return


    pages = data.get(
        "pages",
        []
    )

    target_page = None

    for page in pages:

        if page.get(
            "page_number"
        ) == page_number:

            target_page = page
            break


    if target_page is None:

        print(
            f"\n[WARNING] Page {page_number} "
            f"was not found."
        )

        return


    print("\n")
    print("=" * 80)
    print(
        f"CASE PAGE: {case_id}"
    )
    print(
        f"PAGE: {page_number}"
    )
    print("=" * 80)

    print(
        target_page.get(
            "cleaned_text",
            ""
        )
    )

    print("=" * 80)


def display_supporting_chunks(case):

    chunks = case.get(
        "supporting_chunks",
        []
    )

    if not chunks:
        print(
            "No supporting chunks available."
        )
        return


    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"Evidence {index}"
        )

        print(
            f"Retriever      : "
            f"{chunk.get('retriever')}"
        )

        print(
            f"Retriever rank : "
            f"{chunk.get('rank')}"
        )

        print(
            f"Pages          : "
            f"{chunk.get('pages')}"
        )

        print(
            f"Paragraphs     : "
            f"{chunk.get('paragraph_start')}"
            f"-"
            f"{chunk.get('paragraph_end')}"
        )

        print(
            "\nText:"
        )

        print(
            chunk.get(
                "text",
                ""
            )
        )


def get_existing_annotations(query_record):

    ground_truth = query_record.get(
        "ground_truth",
        {}
    )

    relevant_cases = ground_truth.get(
        "relevant_cases",
        []
    )

    return {
        item.get("case_id")
        for item in relevant_cases
    }


def annotate_candidate(
    query_record,
    candidate,
    candidate_number,
    total_candidates
):

    case_id = candidate.get(
        "case_id",
        ""
    )

    court = candidate.get(
        "court",
        ""
    )

    source_file = candidate.get(
        "source_file",
        ""
    )


    print("\n")
    print("#" * 80)

    print(
        f"CANDIDATE {candidate_number}"
        f" / {total_candidates}"
    )

    print("#" * 80)

    print(
        f"\nCase ID : {case_id}"
    )

    print(
        f"Court   : {court}"
    )

    print(
        f"Source  : {source_file}"
    )

    print(
        f"\nBM25 best rank : "
        f"{candidate.get('bm25_best_rank')}"
    )

    print(
        f"FAISS best rank: "
        f"{candidate.get('faiss_best_rank')}"
    )


    print(
        "\nAvailable commands:"
    )

    print(
        "  d = display supporting evidence"
    )

    print(
        "  p = inspect a page from the actual judgment"
    )

    print(
        "  g = enter relevance grade"
    )

    print(
        "  s = skip this candidate"
    )

    print(
        "  q = quit and save progress"
    )


    while True:

        command = input(
            "\nCommand: "
        ).strip().lower()


        # ----------------------------------------------------
        # DISPLAY EVIDENCE
        # ----------------------------------------------------

        if command == "d":

            display_supporting_chunks(
                candidate
            )

            continue


        # ----------------------------------------------------
        # DISPLAY ACTUAL JUDGMENT PAGE
        # ----------------------------------------------------

        if command == "p":

            page_input = input(
                "Enter page number: "
            ).strip()

            try:

                page_number = int(
                    page_input
                )

            except ValueError:

                print(
                    "Invalid page number."
                )

                continue


            display_case_page(
                case_id,
                page_number
            )

            continue


        # ----------------------------------------------------
        # SKIP
        # ----------------------------------------------------

        if command == "s":

            return None


        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if command == "q":

            return "QUIT"


        # ----------------------------------------------------
        # GRADE
        # ----------------------------------------------------

        if command == "g":

            print(
                "\nRelevance grades:"
            )

            print(
                "3 = Highly relevant"
            )

            print(
                "2 = Relevant"
            )

            print(
                "1 = Partially relevant"
            )

            print(
                "0 = Not relevant"
            )


            grade_input = input(
                "\nEnter grade (0/1/2/3): "
            ).strip()


            if grade_input not in {
                "0",
                "1",
                "2",
                "3"
            }:

                print(
                    "Invalid grade."
                )

                continue


            grade = int(
                grade_input
            )


            # ----------------------------------------------
            # Pages
            # ----------------------------------------------

            pages_input = input(
                "Relevant page(s), e.g. 13,14 "
                "(leave blank if none): "
            ).strip()


            pages = []

            if pages_input:

                for value in pages_input.split(","):

                    value = value.strip()

                    if not value:
                        continue

                    try:

                        pages.append(
                            int(value)
                        )

                    except ValueError:

                        pass


            # ----------------------------------------------
            # Paragraphs
            # ----------------------------------------------

            paragraphs = input(
                "Relevant paragraph(s): "
            ).strip()


            # ----------------------------------------------
            # Reason
            # ----------------------------------------------

            reason = input(
                "Reason for this relevance grade: "
            ).strip()


            annotation = {

                "case_id": case_id,

                "relevance_grade": grade,

                "pages": pages,

                "paragraphs": paragraphs,

                "reason": reason
            }


            print(
                "\nAnnotation recorded:"
            )

            print(
                json.dumps(
                    annotation,
                    indent=4,
                    ensure_ascii=False
                )
            )


            return annotation


        print(
            "Unknown command. "
            "Use d, p, g, s or q."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "INDIAN LEGAL RAG - GROUND TRUTH ANNOTATION"
    )
    print("=" * 80)


    # --------------------------------------------------------
    # Load candidate dataset
    # --------------------------------------------------------

    if not CANDIDATES_FILE.exists():

        print(
            "\nERROR: Candidate file not found:"
        )

        print(
            CANDIDATES_FILE
        )

        return


    data = load_json(
        CANDIDATES_FILE
    )


    queries = data.get(
        "queries",
        []
    )


    print(
        f"\nQueries available: "
        f"{len(queries)}"
    )


    # --------------------------------------------------------
    # Load previous progress if available
    # --------------------------------------------------------

    if OUTPUT_FILE.exists():

        print(
            "\nExisting annotation file found."
        )

        annotated_data = load_json(
            OUTPUT_FILE
        )

        print(
            "Previous progress will be resumed."
        )

    else:

        annotated_data = data

        annotated_data[
            "annotation_status"
        ] = "IN_PROGRESS"


    # --------------------------------------------------------
    # Process each query
    # --------------------------------------------------------

    for query_index, query_record in enumerate(
        annotated_data.get("queries", []),
        start=1
    ):

        query_id = query_record[
            "query_id"
        ]

        query = query_record[
            "query"
        ]

        candidates = query_record.get(
            "candidate_cases",
            []
        )


        # -----------------------------------------------
        # Existing annotations
        # -----------------------------------------------

        ground_truth = query_record.setdefault(
            "ground_truth",
            {}
        )

        relevant_cases = ground_truth.setdefault(
            "relevant_cases",
            []
        )

        annotated_case_ids = (
            get_existing_annotations(
                query_record
            )
        )


        # -----------------------------------------------
        # Display query
        # -----------------------------------------------

        print("\n\n")

        print("=" * 80)

        print(
            f"QUERY {query_index}"
            f" / {len(annotated_data['queries'])}"
        )

        print("=" * 80)

        print(
            f"\nID: {query_id}"
        )

        print(
            f"Domain: "
            f"{query_record.get('domain')}"
        )

        print(
            f"\nQUESTION:\n{query}"
        )

        print(
            f"\nCandidate cases: "
            f"{len(candidates)}"
        )

        print(
            f"Already annotated: "
            f"{len(annotated_case_ids)}"
        )


        # -----------------------------------------------
        # Candidate loop
        # -----------------------------------------------

        for candidate_index, candidate in enumerate(
            candidates,
            start=1
        ):

            case_id = candidate.get(
                "case_id",
                ""
            )

            if case_id in annotated_case_ids:

                continue


            result = annotate_candidate(
                query_record,
                candidate,
                candidate_index,
                len(candidates)
            )


            # -------------------------------------------
            # Quit
            # -------------------------------------------

            if result == "QUIT":

                annotated_data[
                    "annotation_status"
                ] = "IN_PROGRESS"

                save_json(
                    annotated_data,
                    OUTPUT_FILE
                )

                print(
                    "\nProgress saved."
                )

                print(
                    f"File: {OUTPUT_FILE}"
                )

                return


            # -------------------------------------------
            # Skip
            # -------------------------------------------

            if result is None:

                continue


            # -------------------------------------------
            # Store only relevant cases
            # -------------------------------------------

            if result[
                "relevance_grade"
            ] > 0:

                relevant_cases.append(
                    result
                )


            annotated_case_ids.add(
                case_id
            )


            # -------------------------------------------
            # Save after EVERY annotation
            # -------------------------------------------

            ground_truth[
                "relevant_cases"
            ] = relevant_cases

            ground_truth[
                "annotation_status"
            ] = "IN_PROGRESS"


            save_json(
                annotated_data,
                OUTPUT_FILE
            )


            print(
                "\n✓ Annotation saved."
            )


        # ------------------------------------------------
        # Query completed
        # ------------------------------------------------

        ground_truth[
            "annotation_status"
        ] = "COMPLETED"
        
        ground_truth[
            "number_of_relevant_cases"
        ] = len(
            relevant_cases
        )


        save_json(
            annotated_data,
            OUTPUT_FILE
        )


        print(
            "\n✓ Query completed:"
            f" {query_id}"
        )


    # --------------------------------------------------------
    # Entire dataset completed
    # --------------------------------------------------------

    annotated_data[
        "annotation_status"
    ] = "COMPLETED"


    save_json(
        annotated_data,
        OUTPUT_FILE
    )


    print("\n")
    print("=" * 80)
    print(
        "GROUND-TRUTH ANNOTATION COMPLETED"
    )
    print("=" * 80)

    print(
        f"\nOutput file:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()