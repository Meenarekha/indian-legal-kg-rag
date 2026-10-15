import json
import re
import time
from pathlib import Path
from datetime import datetime


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GENERATION_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generation_outputs"
)

CONTEXT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "contexts"
)

REFERENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generation_reference_answers.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generation_evaluations"
)

SUMMARY_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

EVALUATOR_PROVIDER = "gemini"

START_QUERY = 1
END_QUERY = 30

# Set this to True when you want to re-evaluate
# an already evaluated question.
RE_EVALUATE_EXISTING = False

# Small delay between evaluations.
# Helps avoid sending requests too quickly.
DELAY_BETWEEN_QUERIES = 3


# ============================================================
# UTILITIES
# ============================================================

def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_json(path, data):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


def find_latest_file(
    directory,
    pattern
):

    files = list(
        directory.glob(pattern)
    )

    if not files:
        return None

    return max(
        files,
        key=lambda p: p.stat().st_mtime
    )


# ============================================================
# REFERENCE DATA
# ============================================================

def load_reference_dataset():

    data = load_json(
        REFERENCE_FILE
    )

    references = {}

    # --------------------------------------------------------
    # LIST FORMAT
    # --------------------------------------------------------

    if isinstance(data, list):

        for item in data:

            if not isinstance(item, dict):
                continue

            query_id = str(
                item.get("query_id", "")
            ).upper()

            if not query_id:
                continue

            question = (
                item.get("question")
                or item.get("query")
                or ""
            )

            reference_answer = (
                item.get("reference_answer")
                or item.get("answer")
                or ""
            )

            references[query_id] = {

                "question":
                    str(question).strip(),

                "reference_answer":
                    str(reference_answer).strip()
            }

    # --------------------------------------------------------
    # DICTIONARY FORMAT
    # --------------------------------------------------------

    elif isinstance(data, dict):

        # Direct Q001, Q002, etc.
        for key, item in data.items():

            if not str(key).upper().startswith("Q"):
                continue

            if not isinstance(item, dict):
                continue

            query_id = str(
                item.get("query_id")
                or key
            ).upper()

            question = (
                item.get("question")
                or item.get("query")
                or ""
            )

            reference_answer = (
                item.get("reference_answer")
                or item.get("answer")
                or ""
            )

            references[query_id] = {

                "question":
                    str(question).strip(),

                "reference_answer":
                    str(reference_answer).strip()
            }

        # Nested list
        for key in [
            "questions",
            "data",
            "references",
            "items"
        ]:

            value = data.get(key)

            if not isinstance(value, list):
                continue

            for item in value:

                if not isinstance(item, dict):
                    continue

                query_id = str(
                    item.get("query_id", "")
                ).upper()

                if not query_id:
                    continue

                question = (
                    item.get("question")
                    or item.get("query")
                    or ""
                )

                reference_answer = (
                    item.get("reference_answer")
                    or item.get("answer")
                    or ""
                )

                references[query_id] = {

                    "question":
                        str(question).strip(),

                    "reference_answer":
                        str(reference_answer).strip()
                }

    return references


# ============================================================
# LOAD BASE ANSWER
# ============================================================

def load_base_answer(query_id):

    pattern = (
        f"base_answer_{query_id.lower()}_*.json"
    )

    path = find_latest_file(
        GENERATION_OUTPUT_DIR,
        pattern
    )

    if path is None:
        return None

    data = load_json(path)

    answer = str(
        data.get("answer", "")
    ).strip()

    if not answer:
        return None

    return {

        "path":
            str(path),

        "data":
            data,

        "answer":
            answer
    }


# ============================================================
# LOAD RAG ANSWER
# ============================================================

def load_rag_answer(query_id):

    pattern = (
        f"rag_answer_{query_id.lower()}_*.json"
    )

    path = find_latest_file(
        GENERATION_OUTPUT_DIR,
        pattern
    )

    if path is None:
        return None

    data = load_json(path)

    answer = str(
        data.get("answer", "")
    ).strip()

    if not answer:
        return None

    return {

        "path":
            str(path),

        "data":
            data,

        "answer":
            answer
    }


# ============================================================
# FIND RAG CONTEXT
# ============================================================

def find_rag_context(
    rag_data,
    question
):

    # --------------------------------------------------------
    # 1. Explicit context path if available
    # --------------------------------------------------------

    for key in [
        "context_file",
        "context_path",
        "context_package",
        "context_package_path"
    ]:

        value = rag_data.get(key)

        if not value:
            continue

        candidate = Path(
            str(value)
        )

        if candidate.exists():
            return candidate

        candidate = (
            CONTEXT_DIR
            / candidate.name
        )

        if candidate.exists():
            return candidate

    # --------------------------------------------------------
    # 2. Match context filename using question
    # --------------------------------------------------------

    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        question.lower()
    ).strip("_")

    candidates = list(
        CONTEXT_DIR.glob(
            f"context_{normalized[:70]}*.json"
        )
    )

    if candidates:

        return max(
            candidates,
            key=lambda p: p.stat().st_mtime
        )

    return None


# ============================================================
# EXTRACT EVIDENCE
# ============================================================

def extract_evidence(
    context_data
):

    records = context_data.get(
        "evidence_records",
        []
    )

    if not isinstance(records, list):
        return []

    evidence = []

    for index, record in enumerate(
        records,
        start=1
    ):

        if not isinstance(record, dict):
            continue

        evidence_id = (
            record.get("evidence_id")
            or f"E{index:02d}"
        )

        pages = record.get(
            "pages",
            []
        )

        paragraphs = record.get(
            "paragraphs",
            []
        )

        # Support alternate schema.
        if not pages:

            start_page = record.get(
                "start_page"
            )

            end_page = record.get(
                "end_page"
            )

            if start_page:

                if (
                    end_page
                    and end_page != start_page
                ):

                    pages = (
                        f"{start_page}-{end_page}"
                    )

                else:

                    pages = str(
                        start_page
                    )

        if not paragraphs:

            start_para = record.get(
                "paragraph_start"
            )

            end_para = record.get(
                "paragraph_end"
            )

            if start_para:

                if (
                    end_para
                    and end_para != start_para
                ):

                    paragraphs = (
                        f"{start_para}-{end_para}"
                    )

                else:

                    paragraphs = str(
                        start_para
                    )

        evidence.append({

            "evidence_id":
                str(evidence_id),

            "case_id":
                str(
                    record.get(
                        "case_id",
                        "UNKNOWN"
                    )
                ),

            "court":
                str(
                    record.get(
                        "court",
                        "UNKNOWN"
                    )
                ),

            "source_file":
                str(
                    record.get(
                        "source_file",
                        "UNKNOWN"
                    )
                ),

            "pages":
                pages,

            "paragraphs":
                paragraphs,

            "text":
                str(
                    record.get(
                        "text",
                        ""
                    )
                ).strip()
        })

    return evidence


# ============================================================
# FORMAT EVIDENCE
# ============================================================

def format_evidence(
    evidence
):

    if not evidence:

        return (
            "NO RETRIEVED JUDICIAL EVIDENCE "
            "WAS AVAILABLE."
        )

    blocks = []

    for item in evidence:

        block = f"""
[EVIDENCE {item['evidence_id']}]

Case ID:
{item['case_id']}

Court:
{item['court']}

Source File:
{item['source_file']}

Pages:
{item['pages']}

Paragraphs:
{item['paragraphs']}

Judicial Passage:
{item['text']}
"""

        blocks.append(
            block.strip()
        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# EVALUATION PROMPT
# ============================================================

def build_prompt(
    question,
    reference_answer,
    base_answer,
    rag_answer,
    evidence_text
):

    return f"""
You are an automated evaluator for a research prototype
for Indian legal question answering.

IMPORTANT:

- You are NOT a lawyer.
- You are NOT providing professional legal validation.
- Evaluate only the supplied material.
- Do not introduce outside legal facts.
- The reference answer is an evaluation reference.
- The judicial passages are the evidence supplied to the RAG system.
- Do not assume an unsupported claim is correct simply because it
  may be generally known.
- Groundedness measures evidence support, not legal correctness.

============================================================
QUESTION
============================================================

{question}

============================================================
REFERENCE ANSWER
============================================================

{reference_answer}

============================================================
BASE GEMINI ANSWER
============================================================

{base_answer}

============================================================
RAG GEMINI ANSWER
============================================================

{rag_answer}

============================================================
RETRIEVED JUDICIAL EVIDENCE
============================================================

{evidence_text}

============================================================
EVALUATION
============================================================

Evaluate the BASE answer using:

1. relevance
2. reference_consistency
3. completeness

Evaluate the RAG answer using:

1. relevance
2. reference_consistency
3. completeness
4. groundedness

SCORING:

1 = Very poor
2 = Poor
3 = Moderate
4 = Good
5 = Very good

RELEVANCE:
Does the answer directly address the question?

REFERENCE CONSISTENCY:
How closely does the answer agree with the important legal
principles contained in the supplied reference answer?

COMPLETENESS:
How well does the answer cover the important points needed
to answer the question?

GROUNDEDNESS:
How strongly are the legal/factual claims in the RAG answer
supported by the supplied judicial evidence?

IMPORTANT:
A statement should receive grounding credit only when the
supplied evidence supports it.

============================================================
CLAIM-LEVEL GROUNDING
============================================================

For the RAG answer:

- identify meaningful legal/factual claims
- determine whether each is supported by the supplied evidence
- count supported claims
- count unsupported claims
- count total claims

Do not count disclaimers or purely stylistic statements.

============================================================
OVERALL
============================================================

BASE overall =
average of relevance, reference_consistency and completeness.

RAG overall =
average of relevance, reference_consistency,
completeness and groundedness.

Round overall scores to two decimal places.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly:

{{
  "base": {{
    "relevance": 1,
    "reference_consistency": 1,
    "completeness": 1,
    "overall": 1,
    "reason": "short explanation"
  }},

  "rag": {{
    "relevance": 1,
    "reference_consistency": 1,
    "completeness": 1,
    "groundedness": 1,
    "overall": 1,
    "reason": "short explanation"
  }},

  "grounding_analysis": {{
    "supported_claims": 0,
    "unsupported_claims": 0,
    "total_claims": 0,
    "explanation": "short explanation"
  }}
}}

All scores must be integers from 1 to 5,
except overall which may contain two decimal places.
"""


# ============================================================
# CLEAN JSON RESPONSE
# ============================================================

def clean_json_response(
    text
):

    text = str(
        text
    ).strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    return text.strip()


# ============================================================
# RUN GEMINI EVALUATOR
# ============================================================

def run_evaluator(
    prompt
):

    from llm.provider_manager import ProviderManager

    manager = ProviderManager()

    result = manager.generate(
        prompt=prompt,
        primary_provider=EVALUATOR_PROVIDER,
        use_fallback=False
    )

    if isinstance(
        result,
        str
    ):

        raw = result

    elif isinstance(
        result,
        dict
    ):

        if "text" in result:

            raw = result["text"]

        elif "answer" in result:

            raw = result["answer"]

        elif "response" in result:

            raw = result["response"]

        else:

            raise ValueError(
                "ProviderManager returned an unsupported dictionary."
            )

    else:

        raise TypeError(
            "Unexpected evaluator return type: "
            f"{type(result)}"
        )

    cleaned = clean_json_response(
        raw
    )

    try:

        return json.loads(
            cleaned
        )

    except json.JSONDecodeError as error:

        raise ValueError(
            "Gemini evaluator returned invalid JSON.\n\n"
            f"Parser error:\n{error}\n\n"
            f"Response:\n{raw}"
        )


# ============================================================
# VALIDATE EVALUATION
# ============================================================

def validate_evaluation(
    evaluation
):

    if not isinstance(
        evaluation,
        dict
    ):

        raise ValueError(
            "Evaluation is not a JSON object."
        )

    required = [
        "base",
        "rag",
        "grounding_analysis"
    ]

    for section in required:

        if section not in evaluation:

            raise ValueError(
                f"Missing section: {section}"
            )

    base_keys = [
        "relevance",
        "reference_consistency",
        "completeness",
        "overall",
        "reason"
    ]

    rag_keys = [
        "relevance",
        "reference_consistency",
        "completeness",
        "groundedness",
        "overall",
        "reason"
    ]

    for key in base_keys:

        if key not in evaluation["base"]:

            raise ValueError(
                f"Missing BASE metric: {key}"
            )

    for key in rag_keys:

        if key not in evaluation["rag"]:

            raise ValueError(
                f"Missing RAG metric: {key}"
            )


# ============================================================
# SAVE INDIVIDUAL RESULT
# ============================================================

def save_individual_result(
    query_id,
    question,
    reference_answer,
    base_info,
    rag_info,
    context_path,
    evidence,
    evaluation
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    path = (
        OUTPUT_DIR
        / f"evaluation_{query_id.lower()}_{timestamp}.json"
    )

    result = {

        "evaluation_id":
            query_id,

        "created_at":
            datetime.now().isoformat(),

        "evaluation_method": {

            "type":
                "automated_llm_evaluation",

            "human_evaluation":
                False,

            "legal_expert_validation":
                False,

            "generator_provider":
                "gemini",

            "evaluator_provider":
                EVALUATOR_PROVIDER
        },

        "question":
            question,

        "reference_answer":
            reference_answer,

        "base_answer":
            {
                "source_file":
                    base_info["path"],

                "answer":
                    base_info["answer"]
            },

        "rag_answer":
            {
                "source_file":
                    rag_info["path"],

                "answer":
                    rag_info["answer"]
            },

        "rag_context_package":
            str(context_path),

        "rag_evidence":
            evidence,

        "evaluation":
            evaluation,

        "limitations": [

            "Automated evaluation is not legal-expert validation.",

            "Reference answers are evaluation references derived from judicial principles.",

            "LLM-based evaluation cannot guarantee legal correctness.",

            "The same Gemini model family is currently used for generation and evaluation.",

            "Groundedness measures evidence support rather than professional legal correctness."
        ]
    }

    save_json(
        path,
        result
    )

    return path


# ============================================================
# PROCESS ONE QUERY
# ============================================================

def process_query(
    query_id,
    reference
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"{query_id}"
    )

    print(
        "=" * 70
    )

    question = reference[
        "question"
    ]

    reference_answer = reference[
        "reference_answer"
    ]

    print(
        f"Question: {question}"
    )

    # --------------------------------------------------------
    # Base
    # --------------------------------------------------------

    base_info = load_base_answer(
        query_id
    )

    if base_info is None:

        print(
            "\nBASE ANSWER: MISSING"
        )

        return {
            "query_id":
                query_id,

            "status":
                "missing_base_answer"
        }

    print(
        "\nBASE ANSWER: FOUND"
    )

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    rag_info = load_rag_answer(
        query_id
    )

    if rag_info is None:

        print(
            "RAG ANSWER: MISSING"
        )

        return {
            "query_id":
                query_id,

            "status":
                "missing_rag_answer"
        }

    print(
        "RAG ANSWER: FOUND"
    )

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    context_path = find_rag_context(
        rag_info["data"],
        question
    )

    if context_path is None:

        print(
            "RAG CONTEXT: MISSING"
        )

        return {
            "query_id":
                query_id,

            "status":
                "missing_context"
        }

    print(
        f"RAG CONTEXT: {context_path}"
    )

    context_data = load_json(
        context_path
    )

    evidence = extract_evidence(
        context_data
    )

    print(
        f"EVIDENCE RECORDS: {len(evidence)}"
    )

    evidence_text = format_evidence(
        evidence
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = build_prompt(
        question=question,
        reference_answer=reference_answer,
        base_answer=base_info["answer"],
        rag_answer=rag_info["answer"],
        evidence_text=evidence_text
    )

    print(
        "\nRunning automated evaluation..."
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    evaluation = run_evaluator(
        prompt
    )

    validate_evaluation(
        evaluation
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = save_individual_result(
        query_id=query_id,
        question=question,
        reference_answer=reference_answer,
        base_info=base_info,
        rag_info=rag_info,
        context_path=context_path,
        evidence=evidence,
        evaluation=evaluation
    )

    print(
        "\nEvaluation saved:"
    )

    print(
        output_path
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        "\nBASE:"
    )

    print(
        f"  Relevance: "
        f"{evaluation['base']['relevance']}/5"
    )

    print(
        f"  Reference consistency: "
        f"{evaluation['base']['reference_consistency']}/5"
    )

    print(
        f"  Completeness: "
        f"{evaluation['base']['completeness']}/5"
    )

    print(
        f"  Overall: "
        f"{evaluation['base']['overall']}/5"
    )

    print(
        "\nRAG:"
    )

    print(
        f"  Relevance: "
        f"{evaluation['rag']['relevance']}/5"
    )

    print(
        f"  Reference consistency: "
        f"{evaluation['rag']['reference_consistency']}/5"
    )

    print(
        f"  Completeness: "
        f"{evaluation['rag']['completeness']}/5"
    )

    print(
        f"  Groundedness: "
        f"{evaluation['rag']['groundedness']}/5"
    )

    print(
        f"  Overall: "
        f"{evaluation['rag']['overall']}/5"
    )

    grounding = evaluation[
        "grounding_analysis"
    ]

    print(
        "\nGROUNDING:"
    )

    print(
        f"  Supported: "
        f"{grounding['supported_claims']}"
    )

    print(
        f"  Unsupported: "
        f"{grounding['unsupported_claims']}"
    )

    print(
        f"  Total: "
        f"{grounding['total_claims']}"
    )

    return {

        "query_id":
            query_id,

        "status":
            "success",

        "evaluation_file":
            str(output_path),

        "base":
            evaluation["base"],

        "rag":
            evaluation["rag"],

        "grounding":
            grounding
    }


# ============================================================
# AGGREGATE RESULTS
# ============================================================

def calculate_average(
    values
):

    if not values:
        return None

    return round(
        sum(values) / len(values),
        3
    )


def build_aggregate_summary(
    results
):

    successful = [
        item
        for item in results
        if item["status"] == "success"
    ]

    base_relevance = [
        item["base"]["relevance"]
        for item in successful
    ]

    base_consistency = [
        item["base"]["reference_consistency"]
        for item in successful
    ]

    base_completeness = [
        item["base"]["completeness"]
        for item in successful
    ]

    base_overall = [
        item["base"]["overall"]
        for item in successful
    ]

    rag_relevance = [
        item["rag"]["relevance"]
        for item in successful
    ]

    rag_consistency = [
        item["rag"]["reference_consistency"]
        for item in successful
    ]

    rag_completeness = [
        item["rag"]["completeness"]
        for item in successful
    ]

    rag_groundedness = [
        item["rag"]["groundedness"]
        for item in successful
    ]

    rag_overall = [
        item["rag"]["overall"]
        for item in successful
    ]

    supported_claims = sum(
        item["grounding"]["supported_claims"]
        for item in successful
    )

    unsupported_claims = sum(
        item["grounding"]["unsupported_claims"]
        for item in successful
    )

    total_claims = sum(
        item["grounding"]["total_claims"]
        for item in successful
    )

    grounding_percentage = None

    if total_claims > 0:

        grounding_percentage = round(
            (
                supported_claims
                / total_claims
            ) * 100,
            2
        )

    return {

        "created_at":
            datetime.now().isoformat(),

        "evaluation_scope": {

            "start_query":
                START_QUERY,

            "end_query":
                END_QUERY,

            "total_queries":
                END_QUERY - START_QUERY + 1,

            "successful":
                len(successful),

            "failed_or_missing":
                len(results) - len(successful)
        },

        "base_llm_average": {

            "relevance":
                calculate_average(
                    base_relevance
                ),

            "reference_consistency":
                calculate_average(
                    base_consistency
                ),

            "completeness":
                calculate_average(
                    base_completeness
                ),

            "overall":
                calculate_average(
                    base_overall
                )
        },

        "rag_average": {

            "relevance":
                calculate_average(
                    rag_relevance
                ),

            "reference_consistency":
                calculate_average(
                    rag_consistency
                ),

            "completeness":
                calculate_average(
                    rag_completeness
                ),

            "groundedness":
                calculate_average(
                    rag_groundedness
                ),

            "overall":
                calculate_average(
                    rag_overall
                )
        },

        "grounding": {

            "supported_claims":
                supported_claims,

            "unsupported_claims":
                unsupported_claims,

            "total_claims":
                total_claims,

            "support_percentage":
                grounding_percentage
        },

        "query_results":
            results
    }


# ============================================================
# DISPLAY AGGREGATE
# ============================================================

def display_aggregate(
    summary
):

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "GENERATION EVALUATION AGGREGATE"
    )

    print(
        "=" * 70
    )

    scope = summary[
        "evaluation_scope"
    ]

    print(
        "\nEvaluation scope:"
    )

    print(
        f"  Total queries      : "
        f"{scope['total_queries']}"
    )

    print(
        f"  Successfully scored: "
        f"{scope['successful']}"
    )

    print(
        f"  Missing/failed     : "
        f"{scope['failed_or_missing']}"
    )

    print(
        "\nBASE GEMINI"
    )

    print(
        "-" * 70
    )

    for key, value in summary[
        "base_llm_average"
    ].items():

        print(
            f"{key:24}: "
            f"{value}/5"
        )

    print(
        "\nRAG GEMINI"
    )

    print(
        "-" * 70
    )

    for key, value in summary[
        "rag_average"
    ].items():

        print(
            f"{key:24}: "
            f"{value}/5"
        )

    grounding = summary[
        "grounding"
    ]

    print(
        "\nGROUNDING"
    )

    print(
        "-" * 70
    )

    print(
        f"Supported claims       : "
        f"{grounding['supported_claims']}"
    )

    print(
        f"Unsupported claims     : "
        f"{grounding['unsupported_claims']}"
    )

    print(
        f"Total claims           : "
        f"{grounding['total_claims']}"
    )

    print(
        f"Evidence support       : "
        f"{grounding['support_percentage']}%"
    )

    print(
        "=" * 70
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "STEP 16B - AUTOMATED GENERATION EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nLoading reference dataset..."
    )

    references = load_reference_dataset()

    print(
        f"Reference questions loaded: "
        f"{len(references)}"
    )

    results = []

    # --------------------------------------------------------
    # Process Q001-Q030
    # --------------------------------------------------------

    for number in range(
        START_QUERY,
        END_QUERY + 1
    ):

        query_id = f"Q{number:03d}"

        if query_id not in references:

            print(
                f"\n{query_id}: "
                f"REFERENCE ANSWER MISSING"
            )

            results.append({

                "query_id":
                    query_id,

                "status":
                    "missing_reference"
            })

            continue

        # ----------------------------------------------------
        # Skip already evaluated queries if configured
        # ----------------------------------------------------

        if not RE_EVALUATE_EXISTING:

            existing = find_latest_file(
                OUTPUT_DIR,
                f"evaluation_{query_id.lower()}_*.json"
            )

            if existing is not None:

                print(
                    f"\n{query_id}: "
                    f"existing evaluation found."
                )

                existing_data = load_json(
                    existing
                )

                evaluation = existing_data.get(
                    "evaluation"
                )

                grounding = existing_data.get(
                    "evaluation",
                    {}
                ).get(
                    "grounding_analysis",
                    {}
                )

                if evaluation:

                    results.append({

                        "query_id":
                            query_id,

                        "status":
                            "success",

                        "evaluation_file":
                            str(existing),

                        "base":
                            evaluation["base"],

                        "rag":
                            evaluation["rag"],

                        "grounding":
                            grounding
                    })

                else:

                    results.append({

                        "query_id":
                            query_id,

                        "status":
                            "existing_invalid"
                    })

                continue

        # ----------------------------------------------------
        # Process query
        # ----------------------------------------------------

        try:

            result = process_query(
                query_id,
                references[query_id]
            )

            results.append(
                result
            )

        except Exception as error:

            print(
                f"\nERROR processing {query_id}:"
            )

            print(
                error
            )

            results.append({

                "query_id":
                    query_id,

                "status":
                    "error",

                "error":
                    str(error)
            })

        # ----------------------------------------------------
        # Delay
        # ----------------------------------------------------

        if number < END_QUERY:

            print(
                f"\nWaiting "
                f"{DELAY_BETWEEN_QUERIES} seconds..."
            )

            time.sleep(
                DELAY_BETWEEN_QUERIES
            )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    summary = build_aggregate_summary(
        results
    )

    display_aggregate(
        summary
    )

    # --------------------------------------------------------
    # Save aggregate
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    summary_path = (
        SUMMARY_DIR
        / f"generation_evaluation_summary_{timestamp}.json"
    )

    save_json(
        summary_path,
        summary
    )

    print(
        "\nAggregate summary saved:"
    )

    print(
        summary_path
    )

    print(
        "\nStep 16B completed."
    )


if __name__ == "__main__":
    main()