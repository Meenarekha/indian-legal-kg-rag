
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

GEN_DIR = ROOT / "data" / "evaluation" / "generation_outputs"

OUTPUT = ROOT / "data" / "evaluation" / "groundedness_results.json"


# ============================================================
# ENV
# ============================================================

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError("GROQ_API_KEY missing")


client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "openai/gpt-oss-20b"


# ============================================================
# ONLY FAILED QUESTIONS
# ============================================================

FAILED = [
    "Q003",
    "Q007",
    "Q012",
    "Q015",
    "Q024",
    "Q025",
    "Q027",
    "Q030"
]


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# EXTRACT TEXT
# ============================================================

def get_text(record):

    return (
        record.get("text")
        or record.get("chunk_text")
        or record.get("content")
        or ""
    )


# ============================================================
# BUILD SMALL EVIDENCE
# ============================================================

def build_evidence(records):

    blocks = []

    # IMPORTANT
    # Limit evidence to avoid TPM overflow

    for i, r in enumerate(records[:8], start=1):

        text = get_text(r)

        # truncate each evidence

        if len(text) > 1800:
            text = text[:1800]

        case = (
            r.get("case_name")
            or r.get("case_id")
            or "Unknown"
        )

        blocks.append(
            f"""
Evidence {i}
Case: {case}

{text}
"""
        )

    return "\n".join(blocks)


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):

    if not text:
        raise ValueError("Empty response")

    text = text.strip()

    # direct

    try:
        return json.loads(text)
    except:
        pass

    # fenced

    text = re.sub(
        r"```json|```JSON|```",
        "",
        text
    ).strip()

    try:
        return json.loads(text)
    except:
        pass

    # object search

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:

        candidate = text[start:end+1]

        return json.loads(candidate)

    raise ValueError("No JSON found")


# ============================================================
# EVALUATE
# ============================================================

def evaluate(question, answer, evidence):

    prompt = f"""
Evaluate whether the answer is supported by the retrieved evidence.

Use ONLY the evidence.

Question:
{question}

Answer:
{answer}

Evidence:
{evidence}

Return ONLY this JSON:

{{
"groundedness_score":5,
"supported_claims":0,
"unsupported_claims":0,
"total_claims":0,
"overall_reason":"short"
}}
"""

    response = client.chat.completions.create(

        model=MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0,

        max_tokens=1800
    )

    content = response.choices[0].message.content

    return extract_json(content)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RETRYING FAILED GROUNDEDNESS QUESTIONS")
    print("=" * 70)

    if OUTPUT.exists():
        final = load_json(OUTPUT)
    else:
        final = {
            "results": []
        }

    existing = {
        r["query_id"]: r
        for r in final.get("results", [])
    }

    for qid in FAILED:

        print("\n" + "-" * 70)
        print(qid)

        path = GEN_DIR / f"rag_answer_{qid}.json"

        if not path.exists():

            print("Missing file")
            continue

        data = load_json(path)

        question = data["question"]

        answer = data["answer"]

        records = data.get("evidence_records", [])

        print("Evidence:", len(records))

        evidence = build_evidence(records)

        try:

            result = evaluate(
                question,
                answer,
                evidence
            )

            score = int(result["groundedness_score"])
            sup = int(result["supported_claims"])
            unsup = int(result["unsupported_claims"])
            total = int(result["total_claims"])

            print("Score:", score)
            print("Supported:", sup)
            print("Unsupported:", unsup)

            new_result = {
                "query_id": qid,
                "question": question,
                "groundedness_score": score,
                "supported_claims": sup,
                "unsupported_claims": unsup,
                "total_claims": total,
                "overall_reason": result["overall_reason"],
                "evidence_records": len(records),
                "context_characters": data.get(
                    "context_characters",
                    0
                ),
                "evaluator_model": MODEL
            }

            existing[qid] = new_result

        except Exception as e:

            print("FAILED:", e)

    # sort

    ordered = []

    for i in range(1, 31):

        q = f"Q{i:03d}"

        if q in existing:
            ordered.append(existing[q])

    final["results"] = ordered

    final["questions_evaluated"] = len(ordered)

    if ordered:

        avg = sum(
            x["groundedness_score"]
            for x in ordered
        ) / len(ordered)

        final["average_groundedness"] = round(avg, 3)

        sup = sum(
            x["supported_claims"]
            for x in ordered
        )

        unsup = sum(
            x["unsupported_claims"]
            for x in ordered
        )

        total = sum(
            x["total_claims"]
            for x in ordered
        )

        final["supported_claims"] = sup
        final["unsupported_claims"] = unsup
        final["total_claims"] = total

        final["evidence_support_rate_percent"] = round(
            (sup / total * 100) if total else 0,
            2
        )

    save_json(OUTPUT, final)

    print("\nDONE")
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()