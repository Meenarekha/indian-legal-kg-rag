import json
import re
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

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

QUERY_ID = "Q001"


# ============================================================
# JSON UTILITIES
# ============================================================

def load_json(path):
    """Load JSON from a file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def find_latest_file(directory, pattern):
    """Return newest file matching pattern."""

    files = list(
        directory.glob(pattern)
    )

    if not files:
        raise FileNotFoundError(
            f"No files found matching:\n"
            f"{directory / pattern}"
        )

    return max(
        files,
        key=lambda p: p.stat().st_mtime
    )


# ============================================================
# LOAD BASE ANSWER
# ============================================================

def load_base_answer():

    pattern = (
        f"base_answer_{QUERY_ID.lower()}_*.json"
    )

    path = find_latest_file(
        GENERATION_OUTPUT_DIR,
        pattern
    )

    data = load_json(path)

    answer = str(
        data.get("answer", "")
    ).strip()

    if not answer:
        raise ValueError(
            f"Base answer is empty:\n{path}"
        )

    return {
        "path": str(path),
        "data": data,
        "answer": answer
    }


# ============================================================
# LOAD RAG ANSWER
# ============================================================

def load_rag_answer():

    pattern = (
        f"rag_answer_{QUERY_ID.lower()}_*.json"
    )

    path = find_latest_file(
        GENERATION_OUTPUT_DIR,
        pattern
    )

    data = load_json(path)

    answer = str(
        data.get("answer", "")
    ).strip()

    if not answer:
        raise ValueError(
            f"RAG answer is empty:\n{path}"
        )

    return {
        "path": str(path),
        "data": data,
        "answer": answer
    }


# ============================================================
# LOAD REFERENCE ANSWER
# ============================================================

def find_reference_item(data):

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

            if query_id == QUERY_ID:

                question = (
                    item.get("question")
                    or item.get("query")
                    or ""
                )

                reference = (
                    item.get("reference_answer")
                    or item.get("answer")
                    or ""
                )

                return (
                    str(question).strip(),
                    str(reference).strip()
                )

    # --------------------------------------------------------
    # DICTIONARY FORMAT
    # --------------------------------------------------------

    if isinstance(data, dict):

        # Nested list
        for key in [
            "questions",
            "data",
            "references",
            "items"
        ]:

            value = data.get(key)

            if isinstance(value, list):

                for item in value:

                    if not isinstance(item, dict):
                        continue

                    query_id = str(
                        item.get("query_id", "")
                    ).upper()

                    if query_id == QUERY_ID:

                        question = (
                            item.get("question")
                            or item.get("query")
                            or ""
                        )

                        reference = (
                            item.get("reference_answer")
                            or item.get("answer")
                            or ""
                        )

                        return (
                            str(question).strip(),
                            str(reference).strip()
                        )

        # Direct Q001 key
        if QUERY_ID in data:

            item = data[QUERY_ID]

            if isinstance(item, dict):

                question = (
                    item.get("question")
                    or item.get("query")
                    or ""
                )

                reference = (
                    item.get("reference_answer")
                    or item.get("answer")
                    or ""
                )

                return (
                    str(question).strip(),
                    str(reference).strip()
                )

    raise ValueError(
        f"Could not find {QUERY_ID} "
        f"in reference-answer file."
    )


def load_reference_answer():

    data = load_json(
        REFERENCE_FILE
    )

    question, reference = find_reference_item(
        data
    )

    if not reference:

        raise ValueError(
            "Reference answer is empty."
        )

    return {
        "question": question,
        "reference_answer": reference
    }


# ============================================================
# FIND CONTEXT PACKAGE
# ============================================================

def find_rag_context_file(rag_data):

    # --------------------------------------------------------
    # FIRST: Check whether RAG output directly stores
    # the context-file path.
    # --------------------------------------------------------

    possible_keys = [
        "context_file",
        "context_path",
        "context_package",
        "context_package_path"
    ]

    for key in possible_keys:

        value = rag_data.get(key)

        if not value:
            continue

        candidate = Path(
            str(value)
        )

        if candidate.exists():
            return candidate

        # If only filename was stored
        candidate = (
            CONTEXT_DIR
            / candidate.name
        )

        if candidate.exists():
            return candidate

    # --------------------------------------------------------
    # SECOND: Match context file using the question.
    # --------------------------------------------------------

    question = str(
        rag_data.get("question")
        or rag_data.get("query")
        or ""
    ).strip()

    if question:

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

    # --------------------------------------------------------
    # THIRD: Look for recently created context files.
    #
    # This is used only when Q001 is the current experiment.
    # --------------------------------------------------------

    candidates = list(
        CONTEXT_DIR.glob("context_*.json")
    )

    if not candidates:

        raise FileNotFoundError(
            "No context package was found in:\n"
            f"{CONTEXT_DIR}"
        )

    return max(
        candidates,
        key=lambda p: p.stat().st_mtime
    )


# ============================================================
# EXTRACT EVIDENCE FROM CONTEXT PACKAGE
# ============================================================

def extract_evidence_from_context(
    context_data
):

    evidence_records = context_data.get(
        "evidence_records",
        []
    )

    if not isinstance(
        evidence_records,
        list
    ):

        return []

    evidence = []

    for index, record in enumerate(
        evidence_records,
        start=1
    ):

        if not isinstance(record, dict):
            continue

        evidence_id = (
            record.get("evidence_id")
            or f"E{index:02d}"
        )

        case_id = record.get(
            "case_id",
            "UNKNOWN"
        )

        court = record.get(
            "court",
            "UNKNOWN"
        )

        source_file = record.get(
            "source_file",
            "UNKNOWN"
        )

        pages = record.get(
            "pages",
            []
        )

        paragraphs = record.get(
            "paragraphs",
            []
        )

        text = record.get(
            "text",
            ""
        )

        # Some versions may use start/end page fields.
        if not pages:

            start_page = record.get(
                "start_page"
            )

            end_page = record.get(
                "end_page"
            )

            if start_page:

                if end_page and end_page != start_page:

                    pages = (
                        f"{start_page}-{end_page}"
                    )

                else:

                    pages = str(
                        start_page
                    )

        if not paragraphs:

            paragraph_start = record.get(
                "paragraph_start"
            )

            paragraph_end = record.get(
                "paragraph_end"
            )

            if paragraph_start:

                if (
                    paragraph_end
                    and paragraph_end != paragraph_start
                ):

                    paragraphs = (
                        f"{paragraph_start}-{paragraph_end}"
                    )

                else:

                    paragraphs = str(
                        paragraph_start
                    )

        evidence.append({

            "evidence_id": str(
                evidence_id
            ),

            "case_id": str(
                case_id
            ),

            "court": str(
                court
            ),

            "source_file": str(
                source_file
            ),

            "pages": pages,

            "paragraphs": paragraphs,

            "text": str(
                text
            ).strip()
        })

    return evidence


# ============================================================
# FORMAT EVIDENCE
# ============================================================

def format_evidence_for_evaluator(
    evidence
):

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

    if not blocks:

        return (
            "NO RETRIEVED JUDICIAL EVIDENCE "
            "WAS AVAILABLE."
        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# BUILD EVALUATION PROMPT
# ============================================================

def build_evaluation_prompt(
    question,
    reference_answer,
    base_answer,
    rag_answer,
    evidence_text
):

    return f"""
You are an automated evaluator for a research prototype
for Indian legal question answering.

IMPORTANT LIMITATIONS:
- You are NOT a lawyer.
- You are NOT providing professional legal validation.
- Evaluate only the material supplied below.
- Do not introduce outside legal facts.
- The reference answer is an evaluation reference.
- The judicial passages are the evidence supplied to the RAG system.
- Do not assume that a statement is supported by the evidence merely
  because it may be generally true.
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
EVALUATION TASK
============================================================

Evaluate the BASE and RAG answers.

BASE METRICS:

1. relevance
2. reference_consistency
3. completeness

RAG METRICS:

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

DEFINITIONS:

RELEVANCE:
Does the answer directly address the question?

REFERENCE CONSISTENCY:
How closely does the answer agree with the important legal principles
contained in the supplied reference answer?

COMPLETENESS:
How well does the answer cover the important points required to
answer the question?

GROUNDEDNESS:
How strongly are the RAG answer's claims supported by the supplied
judicial evidence?

IMPORTANT:
A legally correct statement that does not appear in the supplied
judicial evidence should NOT receive evidence-grounding credit.

============================================================
GROUNDING CLAIM ANALYSIS
============================================================

For the RAG answer:

1. Identify its meaningful legal/factual claims.
2. Determine whether each claim is supported by the supplied evidence.
3. Count supported claims.
4. Count unsupported claims.
5. Count total claims.

Do not count purely stylistic statements or disclaimers as claims.

============================================================
OVERALL SCORES
============================================================

BASE overall =
average of:
relevance,
reference_consistency,
completeness

RAG overall =
average of:
relevance,
reference_consistency,
completeness,
groundedness

Round overall scores to two decimal places.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly this structure:

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

All metric values must be between 1 and 5.
"""


# ============================================================
# CLEAN GEMINI JSON
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
# RUN AUTOMATED EVALUATOR
# ============================================================

def evaluate_with_llm(
    question,
    reference_answer,
    base_answer,
    rag_answer,
    evidence_text
):

    from llm.provider_manager import ProviderManager

    # ProviderManager does not take the provider
    # in its constructor.
    manager = ProviderManager()

    prompt = build_evaluation_prompt(
        question=question,
        reference_answer=reference_answer,
        base_answer=base_answer,
        rag_answer=rag_answer,
        evidence_text=evidence_text
    )

    print(
        "\nSending Q001 to automated evaluator..."
    )

    result = manager.generate(
        prompt=prompt,
        primary_provider="gemini",
        use_fallback=False
    )

    # --------------------------------------------------------
    # ProviderManager currently returns the generated
    # response as a STRING.
    #
    # Therefore we must NOT use:
    #
    # raw_response = result["text"]
    #
    # because strings cannot be indexed using "text".
    # --------------------------------------------------------

    if isinstance(result, str):

        raw_response = result

    elif isinstance(result, dict):

        # Defensive support in case ProviderManager
        # is changed later to return a dictionary.

        if "text" in result:

            raw_response = result["text"]

        elif "answer" in result:

            raw_response = result["answer"]

        elif "response" in result:

            raw_response = result["response"]

        else:

            raise ValueError(
                "ProviderManager returned a dictionary, "
                "but no supported response field was found.\n"
                f"Available keys: {list(result.keys())}"
            )

    else:

        raise TypeError(
            "Unexpected ProviderManager return type: "
            f"{type(result)}"
        )

    raw_response = str(
        raw_response
    ).strip()

    if not raw_response:

        raise ValueError(
            "Gemini evaluator returned an empty response."
        )

    print(
        "\nRaw evaluator response:"
    )

    print(
        "-" * 70
    )

    print(
        raw_response
    )

    print(
        "-" * 70
    )

    cleaned = clean_json_response(
        raw_response
    )

    try:

        evaluation = json.loads(
            cleaned
        )

    except json.JSONDecodeError as error:

        print(
            "\nERROR: Evaluator response "
            "was not valid JSON."
        )

        print(
            "\nCleaned response:"
        )

        print(
            cleaned
        )

        raise ValueError(
            "Evaluator did not return valid JSON.\n\n"
            f"Parser error:\n{error}\n\n"
            f"Raw response:\n{raw_response}"
        )

    return evaluation

# ============================================================
# VALIDATE EVALUATION
# ============================================================

def validate_evaluation(
    evaluation
):

    required_sections = [
        "base",
        "rag",
        "grounding_analysis"
    ]

    for section in required_sections:

        if section not in evaluation:

            raise ValueError(
                f"Missing evaluation section: "
                f"{section}"
            )

    base_metrics = [
        "relevance",
        "reference_consistency",
        "completeness",
        "overall",
        "reason"
    ]

    rag_metrics = [
        "relevance",
        "reference_consistency",
        "completeness",
        "groundedness",
        "overall",
        "reason"
    ]

    for metric in base_metrics:

        if metric not in evaluation["base"]:

            raise ValueError(
                f"Missing BASE metric: "
                f"{metric}"
            )

    for metric in rag_metrics:

        if metric not in evaluation["rag"]:

            raise ValueError(
                f"Missing RAG metric: "
                f"{metric}"
            )


# ============================================================
# SAVE EVALUATION
# ============================================================

def save_evaluation(
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

    output_path = (
        OUTPUT_DIR
        / f"evaluation_{QUERY_ID.lower()}_{timestamp}.json"
    )

    result = {

        "evaluation_id": QUERY_ID,

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
                "gemini"
        },

        "question":
            question,

        "reference_answer":
            reference_answer,

        "base_answer": {

            "source_file":
                base_info["path"],

            "answer":
                base_info["answer"]
        },

        "rag_answer": {

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

            "The automated evaluator is not a legal expert.",

            "The reference answer is an evaluation reference derived from judicial principles and is not an independent legal opinion.",

            "Automated LLM evaluation cannot guarantee legal correctness.",

            "The same Gemini model family is currently used for generation and evaluation because alternative providers do not currently have usable API credits.",

            "Groundedness measures support from retrieved evidence and does not by itself establish legal correctness."
        ]
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    return output_path


# ============================================================
# DISPLAY SUMMARY
# ============================================================

def display_summary(
    evaluation
):

    print("\n")

    print(
        "=" * 70
    )

    print(
        "Q001 AUTOMATED EVALUATION SUMMARY"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # BASE
    # --------------------------------------------------------

    print(
        "\nBASE LLM"
    )

    print(
        "-" * 70
    )

    print(
        f"Relevance             : "
        f"{evaluation['base']['relevance']}/5"
    )

    print(
        f"Reference consistency : "
        f"{evaluation['base']['reference_consistency']}/5"
    )

    print(
        f"Completeness          : "
        f"{evaluation['base']['completeness']}/5"
    )

    print(
        f"Overall               : "
        f"{evaluation['base']['overall']}/5"
    )

    print(
        f"Reason                : "
        f"{evaluation['base']['reason']}"
    )

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    print(
        "\nRAG"
    )

    print(
        "-" * 70
    )

    print(
        f"Relevance             : "
        f"{evaluation['rag']['relevance']}/5"
    )

    print(
        f"Reference consistency : "
        f"{evaluation['rag']['reference_consistency']}/5"
    )

    print(
        f"Completeness          : "
        f"{evaluation['rag']['completeness']}/5"
    )

    print(
        f"Groundedness          : "
        f"{evaluation['rag']['groundedness']}/5"
    )

    print(
        f"Overall               : "
        f"{evaluation['rag']['overall']}/5"
    )

    print(
        f"Reason                : "
        f"{evaluation['rag']['reason']}"
    )

    # --------------------------------------------------------
    # GROUNDING
    # --------------------------------------------------------

    print(
        "\nGROUNDING ANALYSIS"
    )

    print(
        "-" * 70
    )

    grounding = evaluation[
        "grounding_analysis"
    ]

    print(
        f"Supported claims     : "
        f"{grounding['supported_claims']}"
    )

    print(
        f"Unsupported claims   : "
        f"{grounding['unsupported_claims']}"
    )

    print(
        f"Total claims         : "
        f"{grounding['total_claims']}"
    )

    print(
        f"Explanation          : "
        f"{grounding['explanation']}"
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
        "STEP 16A - Q001 AUTOMATED GENERATION EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nProject root:"
    )

    print(
        PROJECT_ROOT
    )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print(
        "\n[1/6] Loading reference answer..."
    )

    reference = load_reference_answer()

    question = reference[
        "question"
    ]

    reference_answer = reference[
        "reference_answer"
    ]

    print(
        "Reference loaded."
    )

    print(
        f"Question: {question}"
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    print(
        "\n[2/6] Loading Base Gemini answer..."
    )

    base_info = load_base_answer()

    print(
        "Base answer loaded from:"
    )

    print(
        base_info["path"]
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print(
        "\n[3/6] Loading RAG Gemini answer..."
    )

    rag_info = load_rag_answer()

    print(
        "RAG answer loaded from:"
    )

    print(
        rag_info["path"]
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    print(
        "\n[4/6] Locating RAG context package..."
    )

    context_path = find_rag_context_file(
        rag_info["data"]
    )

    print(
        "Context package:"
    )

    print(
        context_path
    )

    context_data = load_json(
        context_path
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    print(
        "\n[5/6] Loading RAG evidence..."
    )

    evidence = extract_evidence_from_context(
        context_data
    )

    print(
        f"Evidence records loaded: "
        f"{len(evidence)}"
    )

    evidence_text = (
        format_evidence_for_evaluator(
            evidence
        )
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    print(
        "\n[6/6] Running automated evaluation..."
    )

    evaluation = evaluate_with_llm(
        question=question,
        reference_answer=reference_answer,
        base_answer=base_info["answer"],
        rag_answer=rag_info["answer"],
        evidence_text=evidence_text
    )

    validate_evaluation(
        evaluation
    )

    output_path = save_evaluation(
        question=question,
        reference_answer=reference_answer,
        base_info=base_info,
        rag_info=rag_info,
        context_path=context_path,
        evidence=evidence,
        evaluation=evaluation
    )

    display_summary(
        evaluation
    )

    print(
        "\nEvaluation completed successfully."
    )

    print(
        "\nSaved result:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()