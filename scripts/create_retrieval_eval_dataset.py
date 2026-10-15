import json
from pathlib import Path


# ============================================================
# RETRIEVAL EVALUATION DATASET
# ============================================================

OUTPUT_DIR = Path("data/evaluation")
OUTPUT_FILE = OUTPUT_DIR / "retrieval_eval.json"


EVALUATION_QUERIES = [

    {
        "query_id": "Q001",
        "domain": "Bail",
        "query": "What are the conditions for granting anticipatory bail under Section 438 CrPC?"
    },

    {
        "query_id": "Q002",
        "domain": "Bail",
        "query": "What principles govern the exercise of discretion when deciding an application for anticipatory bail?"
    },

    {
        "query_id": "Q003",
        "domain": "Bail",
        "query": "When can an accused seek protection from arrest before being taken into custody?"
    },

    {
        "query_id": "Q004",
        "domain": "Bail",
        "query": "What factors should courts consider while deciding whether to grant bail?"
    },

    {
        "query_id": "Q005",
        "domain": "Bail",
        "query": "What is the relationship between personal liberty and the grant of bail?"
    },

    {
        "query_id": "Q006",
        "domain": "Bail",
        "query": "What limitations can be imposed while granting anticipatory bail?"
    },

    {
        "query_id": "Q007",
        "domain": "Bail",
        "query": "Can anticipatory bail continue for an indefinite period?"
    },

    {
        "query_id": "Q008",
        "domain": "Bail",
        "query": "What circumstances may justify cancellation or denial of bail?"
    },

    {
        "query_id": "Q009",
        "domain": "Arrest",
        "query": "What safeguards must be followed during the arrest of an accused?"
    },

    {
        "query_id": "Q010",
        "domain": "Arrest",
        "query": "What are the legal requirements governing arrest without a warrant?"
    },

    {
        "query_id": "Q011",
        "domain": "Custody",
        "query": "What principles govern police custody and judicial remand of an accused?"
    },

    {
        "query_id": "Q012",
        "domain": "Arrest",
        "query": "What rights does an arrested person have during criminal proceedings?"
    },

    {
        "query_id": "Q013",
        "domain": "Arrest",
        "query": "Under what circumstances can an accused challenge the legality of an arrest?"
    },

    {
        "query_id": "Q014",
        "domain": "Criminal Procedure",
        "query": "What principles govern quashing of criminal proceedings under Section 482 CrPC?"
    },

    {
        "query_id": "Q015",
        "domain": "Criminal Procedure",
        "query": "When can the High Court exercise its inherent powers to quash criminal proceedings?"
    },

    {
        "query_id": "Q016",
        "domain": "Criminal Procedure",
        "query": "What are the limitations on the High Court's power to quash criminal proceedings?"
    },

    {
        "query_id": "Q017",
        "domain": "Criminal Procedure",
        "query": "Can criminal proceedings be quashed when the allegations do not disclose an offence?"
    },

    {
        "query_id": "Q018",
        "domain": "Criminal Procedure",
        "query": "Should a High Court conduct a detailed evidentiary inquiry while deciding a quashing petition?"
    },

    {
        "query_id": "Q019",
        "domain": "Evidence",
        "query": "What principles govern the evidentiary value of statements recorded during criminal investigation?"
    },

    {
        "query_id": "Q020",
        "domain": "Evidence",
        "query": "What factors are considered when assessing the credibility of a witness?"
    },

    {
        "query_id": "Q021",
        "domain": "Evidence",
        "query": "What is the evidentiary value of a statement recorded under Section 164 CrPC?"
    },

    {
        "query_id": "Q022",
        "domain": "Evidence",
        "query": "When can inconsistencies in witness statements affect the prosecution case?"
    },

    {
        "query_id": "Q023",
        "domain": "Special Statute",
        "query": "What conditions govern the grant of bail in offences under the Prevention of Money Laundering Act?"
    },

    {
        "query_id": "Q024",
        "domain": "Special Statute",
        "query": "What are the statutory restrictions on granting bail under the PMLA?"
    },

    {
        "query_id": "Q025",
        "domain": "Special Statute",
        "query": "What principles govern the burden on an accused seeking bail under a special statute?"
    },

    {
        "query_id": "Q026",
        "domain": "Special Statute",
        "query": "What conditions apply to bail in cases involving offences under the NDPS Act?"
    },

    {
        "query_id": "Q027",
        "domain": "Bail",
        "query": "How do statutory restrictions affect the grant of bail for serious offences?"
    },

    {
        "query_id": "Q028",
        "domain": "Bail",
        "query": "What factors should courts consider when deciding bail under a special criminal statute?"
    },

    {
        "query_id": "Q029",
        "domain": "Constitutional Law",
        "query": "How does Article 21 protect personal liberty in criminal proceedings?"
    },

    {
        "query_id": "Q030",
        "domain": "Constitutional Law",
        "query": "How should courts balance personal liberty with the interests of criminal investigation?"
    }
]


def create_dataset():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset = {
        "dataset_name": "Indian Legal RAG Retrieval Evaluation",
        "version": "1.0",
        "description": (
            "Evaluation queries for comparing traditional, BM25, "
            "semantic and proposed hybrid retrieval methods."
        ),
        "corpus_description": {
            "total_cases": 205,
            "supreme_court": 105,
            "high_court": 40,
            "district_court": 60
        },
        "annotation_status": "PENDING",
        "ground_truth_policy": (
            "Relevant cases must be identified from the current "
            "205-case corpus using evidence inspection. "
            "Retrieved candidates must not automatically be "
            "treated as ground truth."
        ),
        "queries": []
    }


    for item in EVALUATION_QUERIES:

        record = {
            "query_id": item["query_id"],
            "domain": item["domain"],
            "query": item["query"],

            "relevant_cases": [],

            "relevant_passages": [],

            "annotation_notes": "",

            "annotation_status": "PENDING"
        }

        dataset["queries"].append(record)


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dataset,
            file,
            indent=4,
            ensure_ascii=False
        )


    print("=" * 70)
    print("RETRIEVAL EVALUATION DATASET")
    print("=" * 70)

    print(
        f"\nQueries created : {len(dataset['queries'])}"
    )

    print(
        f"Output file     : {OUTPUT_FILE}"
    )

    print(
        "\nGround-truth status: PENDING"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Relevant cases must be manually validated "
        "from the current 205-case corpus."
    )

    print(
        "\nDataset creation completed successfully."
    )


if __name__ == "__main__":
    create_dataset()