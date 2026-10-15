import json
import os
import re
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "reference_answers.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generation_outputs"
)

EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evaluation_results"
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv(
    PROJECT_ROOT / ".env"
)


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

if not GROQ_API_KEY:

    raise RuntimeError(
        "GROQ_API_KEY is not set in the .env file."
    )


client = Groq(
    api_key=GROQ_API_KEY
)

# Model confirmed to be available
# for the user's Groq account.
EVALUATOR_MODEL = "openai/gpt-oss-120b"


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# NORMALIZE QUERY ID
# ============================================================

def normalize_query_id(query_id):

    return str(
        query_id
    ).strip().upper()


# ============================================================
# FIND LATEST SUCCESSFUL ANSWERS
# ============================================================

def find_latest_answers(
    generation_type
):

    """
    Finds the latest successful answer for every
    query ID.

    Multiple historical generation files may exist.
    The answer with the latest generated_at timestamp
    is selected.
    """

    answers = {}

    for path in OUTPUT_DIR.glob(
        "*.json"
    ):

        filename = path.name.lower()

        # ----------------------------------------------------
        # Select Base or RAG files
        # ----------------------------------------------------

        if generation_type == "BASE_LLM":

            if not filename.startswith(
                "base_answer_"
            ):

                continue

        elif generation_type == "RAG":

            if not filename.startswith(
                "rag_answer_"
            ):

                continue

        else:

            continue

        # ----------------------------------------------------
        # Read file
        # ----------------------------------------------------

        try:

            data = load_json(
                path
            )

        except Exception:

            continue

        # ----------------------------------------------------
        # Validate answer
        # ----------------------------------------------------

        query_id = data.get(
            "query_id"
        )

        answer = data.get(
            "answer"
        )

        if not query_id:

            continue

        if not answer:

            continue

        query_id = normalize_query_id(
            query_id
        )

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        generated_at = data.get(
            "generated_at",
            ""
        )

        # ----------------------------------------------------
        # Keep latest
        # ----------------------------------------------------

        if query_id not in answers:

            answers[
                query_id
            ] = (
                path,
                data
            )

        else:

            old_path, old_data = (
                answers[query_id]
            )

            old_timestamp = (
                old_data.get(
                    "generated_at",
                    ""
                )
            )

            if generated_at > old_timestamp:

                answers[
                    query_id
                ] = (
                    path,
                    data
                )

    return answers


# ============================================================
# LOAD RAG EVIDENCE
# ============================================================

def load_rag_evidence(
    rag_data
):

    """
    Loads the actual judicial evidence that was
    supplied to the RAG generator.
    """

    context_file = rag_data.get(
        "context_file"
    )

    if not context_file:

        return (
            "No RAG context file was recorded."
        )

    context_path = (
        PROJECT_ROOT
        / context_file
    )

    if not context_path.exists():

        return (
            "RAG context file was not found: "
            f"{context_file}"
        )

    # --------------------------------------------------------
    # Read context
    # --------------------------------------------------------

    try:

        context_data = load_json(
            context_path
        )

    except Exception as error:

        return (
            "Unable to load RAG context file: "
            f"{error}"
        )

    # --------------------------------------------------------
    # Evidence records
    # --------------------------------------------------------

    evidence_records = (
        context_data.get(
            "evidence_records",
            []
        )
    )

    if not evidence_records:

        return (
            "No evidence records were found "
            "in the RAG context."
        )

    evidence_text = []

    for index, record in enumerate(
        evidence_records,
        start=1
    ):

        case_name = record.get(
            "case_name",
            "Unknown case"
        )

        court = record.get(
            "court",
            "Unknown court"
        )

        source = record.get(
            "source",
            ""
        )

        page = record.get(
            "page",
            ""
        )

        paragraph = record.get(
            "paragraph",
            ""
        )

        text = record.get(
            "text",
            ""
        )

        evidence_text.append(
            f"""
EVIDENCE {index}

Case: {case_name}
Court: {court}
Source: {source}
Page: {page}
Paragraph: {paragraph}

Judicial text:
{text}
"""
        )

    combined = "\n".join(
        evidence_text
    )

    # --------------------------------------------------------
    # Limit evaluator prompt size
    # --------------------------------------------------------

    MAX_EVIDENCE_CHARS = 25000

    if len(combined) > MAX_EVIDENCE_CHARS:

        combined = combined[
            :MAX_EVIDENCE_CHARS
        ]

        combined += (
            "\n\n[Evidence truncated for evaluation.]"
        )

    return combined


# ============================================================
# EXTRACT JSON FROM EVALUATOR RESPONSE
# ============================================================

def extract_json(text):

    text = text.strip()

    # --------------------------------------------------------
    # Attempt 1: Entire response is JSON
    # --------------------------------------------------------

    try:

        return json.loads(
            text
        )

    except json.JSONDecodeError:

        pass

    # --------------------------------------------------------
    # Attempt 2: JSON inside markdown code block
    # --------------------------------------------------------

    match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        re.DOTALL
    )

    if match:

        try:

            return json.loads(
                match.group(1)
            )

        except json.JSONDecodeError:

            pass

    # --------------------------------------------------------
    # Attempt 3: Find first JSON object
    # --------------------------------------------------------

    start = text.find(
        "{"
    )

    end = text.rfind(
        "}"
    )

    if (
        start != -1
        and
        end != -1
        and
        end > start
    ):

        candidate = text[
            start:end + 1
        ]

        try:

            return json.loads(
                candidate
            )

        except json.JSONDecodeError:

            pass

    raise ValueError(
        "Evaluator did not return valid JSON.\n"
        f"Raw response:\n{text}"
    )


# ============================================================
# VALIDATE SCORE
# ============================================================

def validate_score(
    value,
    metric_name
):

    try:

        score = float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        raise ValueError(
            f"Invalid score for "
            f"{metric_name}: {value}"
        )

    if score < 1 or score > 5:

        raise ValueError(
            f"Score for {metric_name} "
            f"must be between 1 and 5. "
            f"Received: {score}"
        )

    return score


# ============================================================
# BUILD EVALUATION PROMPT
# ============================================================

def build_evaluation_prompt(
    question,
    reference_answer,
    base_answer,
    rag_answer,
    rag_evidence
):

    return f"""
You are an automated evaluator for an academic
Indian Legal Question-Answering RAG experiment.

IMPORTANT:
You are evaluating generated answers, not providing
legal advice.

Compare the Base LLM answer and RAG answer against
the supplied reference answer.

Do not invent facts.

Do not perform an independent legal analysis beyond
the supplied reference answer and retrieved evidence.

The reference answer is an evaluation reference.
It is not necessarily an exhaustive statement of all law.

============================================================
QUESTION
============================================================

{question}

============================================================
REFERENCE ANSWER
============================================================

{reference_answer}

============================================================
BASE LLM ANSWER
============================================================

{base_answer}

============================================================
RAG ANSWER
============================================================

{rag_answer}

============================================================
RETRIEVED JUDICIAL EVIDENCE USED BY RAG
============================================================

{rag_evidence}

============================================================
SCORING RUBRIC
============================================================

Give every metric an integer score from 1 to 5.

------------------------------------------------------------
RELEVANCE
------------------------------------------------------------

5 = Directly and comprehensively addresses the question.
4 = Clearly addresses the question with minor omissions.
3 = Partially addresses the question.
2 = Mostly off-topic or misses important aspects.
1 = Does not answer the question.

------------------------------------------------------------
REFERENCE CONSISTENCY
------------------------------------------------------------

5 = Consistent with the reference answer with no
    meaningful contradiction.

4 = Mostly consistent with minor differences.

3 = Partially consistent with some important differences.

2 = Significant contradiction or unsupported claims.

1 = Fundamentally inconsistent with the reference.

------------------------------------------------------------
COMPLETENESS
------------------------------------------------------------

5 = Covers essentially all important points in the
    reference answer.

4 = Covers most important points.

3 = Covers some important points but misses several.

2 = Covers only a small portion.

1 = Barely addresses the expected content.

------------------------------------------------------------
GROUNDEDNESS
------------------------------------------------------------

This metric applies ONLY to the RAG answer.

5 = Important claims are strongly supported by the
    supplied retrieved judicial evidence.

4 = Most important claims are supported.

3 = Some important claims are supported, while others
    are weakly supported or not clearly evidenced.

2 = Many claims lack support in the retrieved evidence.

1 = The answer is largely unsupported by the retrieved
    evidence.

IMPORTANT:
Groundedness means support by the supplied retrieved
evidence.

Groundedness does NOT mean professional legal correctness.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Do NOT include markdown.

Do NOT include explanations outside the JSON.

Use exactly this structure:

{{
  "base": {{
    "relevance": 1,
    "reference_consistency": 1,
    "completeness": 1
  }},

  "rag": {{
    "relevance": 1,
    "reference_consistency": 1,
    "completeness": 1,
    "groundedness": 1
  }},

  "brief_reason": "Short explanation of the main difference between the Base and RAG answers."
}}
"""


# ============================================================
# EVALUATE ONE QUESTION
# ============================================================

def evaluate_question(
    question,
    reference_answer,
    base_answer,
    rag_answer,
    rag_evidence
):

    prompt = build_evaluation_prompt(
        question=question,
        reference_answer=reference_answer,
        base_answer=base_answer,
        rag_answer=rag_answer,
        rag_evidence=rag_evidence
    )

    # --------------------------------------------------------
    # Call evaluator
    # --------------------------------------------------------

    response = client.chat.completions.create(

        model=EVALUATOR_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0,

        max_tokens=1500
    )

    # --------------------------------------------------------
    # Validate response
    # --------------------------------------------------------

    if not response.choices:

        raise RuntimeError(
            "Groq evaluator returned no choices."
        )

    text = response.choices[
        0
    ].message.content

    if not text:

        raise RuntimeError(
            "Groq evaluator returned empty content."
        )

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    result = extract_json(
        text
    )

    # ========================================================
    # VALIDATE BASE
    # ========================================================

    if "base" not in result:

        raise ValueError(
            "Evaluator response does not contain 'base'."
        )

    required_base_metrics = [
        "relevance",
        "reference_consistency",
        "completeness"
    ]

    for metric in required_base_metrics:

        if metric not in result["base"]:

            raise ValueError(
                "Evaluator response is missing "
                f"base.{metric}"
            )

        result["base"][metric] = validate_score(
            result["base"][metric],
            f"base.{metric}"
        )

    # ========================================================
    # VALIDATE RAG
    # ========================================================

    if "rag" not in result:

        raise ValueError(
            "Evaluator response does not contain 'rag'."
        )

    required_rag_metrics = [
        "relevance",
        "reference_consistency",
        "completeness",
        "groundedness"
    ]

    for metric in required_rag_metrics:

        if metric not in result["rag"]:

            raise ValueError(
                "Evaluator response is missing "
                f"rag.{metric}"
            )

        result["rag"][metric] = validate_score(
            result["rag"][metric],
            f"rag.{metric}"
        )

    # ========================================================
    # CALCULATE OVERALL SCORES PROGRAMMATICALLY
    # ========================================================

    base_scores = [
        result["base"]["relevance"],
        result["base"]["reference_consistency"],
        result["base"]["completeness"]
    ]

    rag_scores = [
        result["rag"]["relevance"],
        result["rag"]["reference_consistency"],
        result["rag"]["completeness"],
        result["rag"]["groundedness"]
    ]

    result["base"]["overall"] = round(
        sum(base_scores)
        /
        len(base_scores),
        2
    )

    result["rag"]["overall"] = round(
        sum(rag_scores)
        /
        len(rag_scores),
        2
    )

    return result


# ============================================================
# SAVE INDIVIDUAL RESULT
# ============================================================

def save_result(
    result
):

    path = (
        EVALUATION_DIR
        /
        f"evaluation_{result['query_id']}.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False
        )

    return path


# ============================================================
# CALCULATE SUMMARY
# ============================================================

def calculate_summary(
    results
):

    base_results = [
        item["base"]
        for item in results
    ]

    rag_results = [
        item["rag"]
        for item in results
    ]

    # --------------------------------------------------------
    # Average helper
    # --------------------------------------------------------

    def average(values):

        if not values:

            return 0

        return round(
            sum(values)
            /
            len(values),
            3
        )

    # ========================================================
    # BASE
    # ========================================================

    base_relevance = average(
        [
            x["relevance"]
            for x in base_results
        ]
    )

    base_consistency = average(
        [
            x["reference_consistency"]
            for x in base_results
        ]
    )

    base_completeness = average(
        [
            x["completeness"]
            for x in base_results
        ]
    )

    base_overall = average(
        [
            x["overall"]
            for x in base_results
        ]
    )

    # ========================================================
    # RAG
    # ========================================================

    rag_relevance = average(
        [
            x["relevance"]
            for x in rag_results
        ]
    )

    rag_consistency = average(
        [
            x["reference_consistency"]
            for x in rag_results
        ]
    )

    rag_completeness = average(
        [
            x["completeness"]
            for x in rag_results
        ]
    )

    rag_groundedness = average(
        [
            x["groundedness"]
            for x in rag_results
        ]
    )

    rag_overall = average(
        [
            x["overall"]
            for x in rag_results
        ]
    )

    # ========================================================
    # DIFFERENCE
    # ========================================================

    difference = {

        "overall":
            round(
                rag_overall
                -
                base_overall,
                3
            ),

        "relevance":
            round(
                rag_relevance
                -
                base_relevance,
                3
            ),

        "reference_consistency":
            round(
                rag_consistency
                -
                base_consistency,
                3
            ),

        "completeness":
            round(
                rag_completeness
                -
                base_completeness,
                3
            )
    }

    # ========================================================
    # IMPROVEMENT PERCENTAGE
    # ========================================================

    def improvement_percentage(
        base,
        rag
    ):

        if base == 0:

            return None

        return round(
            (
                (rag - base)
                /
                base
            )
            *
            100,
            2
        )

    improvement = {

        "overall_percent":
            improvement_percentage(
                base_overall,
                rag_overall
            ),

        "relevance_percent":
            improvement_percentage(
                base_relevance,
                rag_relevance
            ),

        "reference_consistency_percent":
            improvement_percentage(
                base_consistency,
                rag_consistency
            ),

        "completeness_percent":
            improvement_percentage(
                base_completeness,
                rag_completeness
            )
    }

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    return {

        "number_of_questions":
            len(results),

        "base_llm": {

            "relevance":
                base_relevance,

            "reference_consistency":
                base_consistency,

            "completeness":
                base_completeness,

            "overall":
                base_overall
        },

        "rag": {

            "relevance":
                rag_relevance,

            "reference_consistency":
                rag_consistency,

            "completeness":
                rag_completeness,

            "groundedness":
                rag_groundedness,

            "overall":
                rag_overall
        },

        "difference_rag_minus_base":
            difference,

        "improvement_percentage":
            improvement,

        "evaluation_method":
            "Automated LLM-based evaluation",

        "evaluator_model":
            EVALUATOR_MODEL,

        "legal_expert_validation":
            False,

        "reference_answers_used_during_generation":
            False,

        "retrieval_used_during_rag_generation":
            True,

        "note":
            "Scores measure alignment with the prepared "
            "evaluation references and support from retrieved "
            "judicial evidence. They do not constitute "
            "professional legal validation or certification."
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("AUTOMATED BASE VS RAG EVALUATION")
    print("=" * 80)

    # ========================================================
    # LOAD REFERENCE ANSWERS
    # ========================================================

    reference_data = load_json(
        REFERENCE_FILE
    )

    reference_questions = {}

    for item in reference_data.get(
        "answers",
        []
    ):

        query_id = normalize_query_id(
            item["query_id"]
        )

        reference_questions[
            query_id
        ] = item

    # ========================================================
    # FIND ANSWERS
    # ========================================================

    base_answers = find_latest_answers(
        "BASE_LLM"
    )

    rag_answers = find_latest_answers(
        "RAG"
    )

    print(
        f"\nReference questions: "
        f"{len(reference_questions)}"
    )

    print(
        f"Base answers found: "
        f"{len(base_answers)}"
    )

    print(
        f"RAG answers found: "
        f"{len(rag_answers)}"
    )

    # ========================================================
    # EVALUATION LOOP
    # ========================================================

    results = []

    for number in range(
        1,
        31
    ):

        query_id = (
            f"Q{number:03d}"
        )

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        existing_result = (
            EVALUATION_DIR
            /
            f"evaluation_{query_id}.json"
        )

        if existing_result.exists():

            print(
                "\n"
                +
                "-" * 80
            )

            print(
                f"Skipping {query_id}: "
                "evaluation already exists."
            )

            # Load existing result so it contributes
            # to the final summary.
            try:

                existing_data = load_json(
                    existing_result
                )

                if (
                    "base" in existing_data
                    and
                    "rag" in existing_data
                    and
                    "overall" in existing_data["base"]
                    and
                    "overall" in existing_data["rag"]
                ):

                    results.append(
                        existing_data
                    )

                else:

                    print(
                        f"Existing result for "
                        f"{query_id} is incomplete. "
                        "It will be re-evaluated."
                    )

            except Exception as error:

                print(
                    f"Could not load existing "
                    f"result for {query_id}: "
                    f"{error}"
                )

            continue

        # ----------------------------------------------------
        # START EVALUATION
        # ----------------------------------------------------

        print(
            "\n"
            +
            "-" * 80
        )

        print(
            f"Evaluating {query_id}"
        )

        # ----------------------------------------------------
        # Check reference
        # ----------------------------------------------------

        if query_id not in reference_questions:

            print(
                f"SKIPPED {query_id}: "
                "reference answer missing."
            )

            continue

        # ----------------------------------------------------
        # Check Base
        # ----------------------------------------------------

        if query_id not in base_answers:

            print(
                f"SKIPPED {query_id}: "
                "Base answer missing."
            )

            continue

        # ----------------------------------------------------
        # Check RAG
        # ----------------------------------------------------

        if query_id not in rag_answers:

            print(
                f"SKIPPED {query_id}: "
                "RAG answer missing."
            )

            continue

        # ====================================================
        # GET DATA
        # ====================================================

        reference = reference_questions[
            query_id
        ]

        base_path, base_data = (
            base_answers[
                query_id
            ]
        )

        rag_path, rag_data = (
            rag_answers[
                query_id
            ]
        )

        question = reference[
            "question"
        ]

        reference_answer = reference[
            "reference_answer"
        ]

        base_answer = base_data[
            "answer"
        ]

        rag_answer = rag_data[
            "answer"
        ]

        # ====================================================
        # LOAD RAG EVIDENCE
        # ====================================================

        rag_evidence = load_rag_evidence(
            rag_data
        )

        # ====================================================
        # RUN EVALUATION
        # ====================================================

        try:

            scores = evaluate_question(

                question=
                    question,

                reference_answer=
                    reference_answer,

                base_answer=
                    base_answer,

                rag_answer=
                    rag_answer,

                rag_evidence=
                    rag_evidence
            )

        except Exception as error:

            print(
                "\nEVALUATION FAILED:"
            )

            print(
                error
            )

            # Continue to next question
            # rather than crashing the entire run.
            continue

        # ====================================================
        # BUILD RESULT
        # ====================================================

        result = {

            "query_id":
                query_id,

            "question":
                question,

            "base_source_file":
                str(
                    base_path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "rag_source_file":
                str(
                    rag_path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "base_provider":
                base_data.get(
                    "provider"
                ),

            "rag_provider":
                rag_data.get(
                    "provider"
                ),

            "evaluation_model":
                EVALUATOR_MODEL,

            "evaluation_timestamp":
                datetime.now().isoformat(),

            "base":
                scores["base"],

            "rag":
                scores["rag"],

            "brief_reason":
                scores.get(
                    "brief_reason",
                    ""
                )
        }

        # ====================================================
        # SAVE IMMEDIATELY
        # ====================================================

        save_result(
            result
        )

        results.append(
            result
        )

        # ====================================================
        # PRINT RESULT
        # ====================================================

        print(
            f"Base overall: "
            f"{result['base']['overall']}"
        )

        print(
            f"RAG overall: "
            f"{result['rag']['overall']}"
        )

        print(
            "Saved successfully."
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")

    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    if not results:

        print(
            "No evaluations were completed."
        )

        return

    # --------------------------------------------------------
    # Sort results by QID
    # --------------------------------------------------------

    results.sort(
        key=lambda x:
            int(
                x["query_id"][1:]
            )
    )

    # --------------------------------------------------------
    # Calculate summary
    # --------------------------------------------------------

    summary = calculate_summary(
        results
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = (
        EVALUATION_DIR
        /
        "evaluation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print(
        f"\nQuestions evaluated: "
        f"{summary['number_of_questions']}"
    )

    print("\nBASE LLM")

    print(
        f"Relevance: "
        f"{summary['base_llm']['relevance']}"
    )

    print(
        f"Reference consistency: "
        f"{summary['base_llm']['reference_consistency']}"
    )

    print(
        f"Completeness: "
        f"{summary['base_llm']['completeness']}"
    )

    print(
        f"Overall: "
        f"{summary['base_llm']['overall']}"
    )

    print("\nRAG")

    print(
        f"Relevance: "
        f"{summary['rag']['relevance']}"
    )

    print(
        f"Reference consistency: "
        f"{summary['rag']['reference_consistency']}"
    )

    print(
        f"Completeness: "
        f"{summary['rag']['completeness']}"
    )

    print(
        f"Groundedness: "
        f"{summary['rag']['groundedness']}"
    )

    print(
        f"Overall: "
        f"{summary['rag']['overall']}"
    )

    print(
        "\nRAG - BASE DIFFERENCE"
    )

    print(
        f"Overall: "
        f"{summary['difference_rag_minus_base']['overall']}"
    )

    print(
        f"Relevance: "
        f"{summary['difference_rag_minus_base']['relevance']}"
    )

    print(
        f"Reference consistency: "
        f"{summary['difference_rag_minus_base']['reference_consistency']}"
    )

    print(
        f"Completeness: "
        f"{summary['difference_rag_minus_base']['completeness']}"
    )

    print(
        "\nIMPROVEMENT PERCENTAGE"
    )

    print(
        f"Overall: "
        f"{summary['improvement_percentage']['overall_percent']}%"
    )

    print(
        f"Relevance: "
        f"{summary['improvement_percentage']['relevance_percent']}%"
    )

    print(
        f"Reference consistency: "
        f"{summary['improvement_percentage']['reference_consistency_percent']}%"
    )

    print(
        f"Completeness: "
        f"{summary['improvement_percentage']['completeness_percent']}%"
    )

    print(
        "\nSummary saved to:"
    )

    print(
        summary_path
    )

    print(
        "\nEvaluation complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()