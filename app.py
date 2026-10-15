import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import re
import sys
import json
import html as html_lib
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Legal Query RAG")
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
KG_FILE = PROJECT_ROOT / "knowledge_graph" / "data" / "nodes" / "all_nodes.json"

for path in (SCRIPTS_DIR, PROJECT_ROOT / "knowledge_graph" / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

st.set_page_config(
    page_title="Indian Legal KG + RAG",
    page_icon="⚖️",
    layout="wide",
)

# ------------------------------------------------------------
# LOAD EXISTING RAG COMPONENTS AND KNOWLEDGE GRAPH
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_rag_system():
    from context_builder import load_retrieval_system
    return load_retrieval_system()


@st.cache_resource(show_spinner=False)
def load_knowledge_graph():
    if not KG_FILE.exists():
        raise FileNotFoundError(f"Knowledge Graph file not found: {KG_FILE}")
    with open(KG_FILE, "r", encoding="utf-8") as f:
        graph = json.load(f)
    return graph, {n.get("node_id"): n for n in graph.get("nodes", []) if n.get("node_id")}


def normalize(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def extract_structured_query_entities(query):
    entities = []
    for match in re.finditer(r"\bsection\s+(\d+[a-z]?)\b", query, re.I):
        entities.append(("Section", match.group(1)))
    for match in re.finditer(r"\barticle\s+(\d+[a-z]?)\b", query, re.I):
        entities.append(("Article", match.group(1)))
    # preserve order, remove duplicate matches
    return list(dict.fromkeys(entities))


def find_kg_nodes(graph, query_entities):
    matched = []
    nodes = graph.get("nodes", [])
    for entity_type, number in query_entities:
        target_number = normalize(number)
        for node in nodes:
            node_type = str(node.get("node_type", node.get("type", ""))).lower()
            props = node.get("properties", {}) or {}
            node_id = normalize(node.get("node_id", ""))
            name_values = [
                props.get("section_name", ""),
                props.get("article_name", ""),
                props.get("name", ""),
                props.get("label", ""),
            ]
            # This KG stores constitutional Article entities using the Section node family too.
            type_ok = (
                entity_type.lower() in node_type
                or (entity_type == "Article" and "section" in node_type)
                or (entity_type == "Section" and "section" in node_type)
            )
            name_match = any(
                target_number == normalize(re.search(r"\d+[a-z]?", str(v), re.I).group(0))
                for v in name_values
                if v and re.search(r"\d+[a-z]?", str(v), re.I)
            )
            id_match = target_number in node_id and (
                "section" in node_id or "article" in node_id
            )
            if type_ok and (name_match or id_match):
                matched.append(node)
    # unique node ids
    unique = {}
    for node in matched:
        unique[node.get("node_id")] = node
    return list(unique.values())


def get_kg_case_candidates(graph, node_lookup, matched_nodes):
    matched_node_ids = {n.get("node_id") for n in matched_nodes}
    candidates = set()
    relationships = graph.get("relationships", [])
    for rel in relationships:
        rel_type = str(rel.get("relationship_type", rel.get("type", ""))).upper()
        if rel_type != "INVOLVES_SECTION":
            continue
        if rel.get("target") not in matched_node_ids:
            continue
        case_id = rel.get("source")
        case_node = node_lookup.get(case_id, {})
        props = case_node.get("properties", {}) or {}
        if case_node and str(case_node.get("node_type", case_node.get("type", ""))).lower() != "case":
            continue
        if props.get("reference_only", False):
            continue
        if case_id:
            candidates.add(case_id)
    return candidates


def make_candidate_specific_results(query, bm25, bm25_metadata, faiss_index, faiss_metadata, model, candidate_ids):
    """Re-score chunks belonging to the union of KG and RAG candidate cases."""
    from hybrid_search import tokenize, create_result, reciprocal_rank_fusion
    from case_level_search import aggregate_by_case
    from authority_rerank import relevance_first_filter, authority_aware_rerank

    candidate_ids = set(candidate_ids)
    if not candidate_ids:
        return [], [], []

    # BM25: score corpus, then retain chunks from candidate cases.
    bm25_scores = np.asarray(bm25.get_scores(tokenize(query)))
    bm25_rows = [
        (int(i), float(bm25_scores[i]))
        for i, meta in enumerate(bm25_metadata)
        if meta.get("case_id") in candidate_ids
    ]
    bm25_rows.sort(key=lambda row: row[1], reverse=True)
    bm25_results = [
        create_result(bm25_metadata, idx, rank, "bm25", score)
        for rank, (idx, score) in enumerate(bm25_rows, start=1)
    ]

    # Dense retrieval: score the complete FAISS index, then retain candidate cases.
    query_embedding = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    dense_scores, dense_indices = faiss_index.search(query_embedding, int(faiss_index.ntotal))
    dense_rows = []
    for score, idx in zip(dense_scores[0], dense_indices[0]):
        idx = int(idx)
        if idx < 0 or idx >= len(faiss_metadata):
            continue
        if faiss_metadata[idx].get("case_id") in candidate_ids:
            dense_rows.append((idx, float(score)))
    # FAISS already returns descending similarity; sort explicitly for safety.
    dense_rows.sort(key=lambda row: row[1], reverse=True)
    faiss_results = [
        create_result(faiss_metadata, idx, rank, "faiss", score)
        for rank, (idx, score) in enumerate(dense_rows, start=1)
    ]

    fused = reciprocal_rank_fusion(bm25_results, faiss_results, 60)
    case_results = aggregate_by_case(fused)
    filtered_cases = relevance_first_filter(case_results)
    reranked_cases = authority_aware_rerank(filtered_cases)

    # KG contributes to retrieval in two explicit ways: candidate generation and a small,
    # bounded reranking bonus. It cannot override a poor retrieval score by itself.
    for case in reranked_cases:
        is_kg = case.get("case_id") in candidate_ids and case.get("case_id") in getattr(st.session_state, "kg_case_ids", set())
        case["kg_match"] = bool(is_kg)
        case["kg_score"] = 1.0 if is_kg else 0.0
        old_score = float(case.get("final_score", 0.0) or 0.0)
        case["final_score_before_kg"] = old_score
        case["final_score"] = (0.95 * old_score) + (0.05 * case["kg_score"])
    reranked_cases.sort(key=lambda case: float(case.get("final_score", 0.0) or 0.0), reverse=True)
    return case_results, filtered_cases, reranked_cases


# ------------------------------------------------------------
# UI HELPERS
# ------------------------------------------------------------
def esc(value):
    return html_lib.escape(str(value))


def render_html(markup):
    """Render HTML without markdown treating indented lines as a code block."""
    st.markdown(re.sub(r"\n\s+", "\n", markup).strip(), unsafe_allow_html=True)


def set_example_query(text):
    st.session_state["legal_query"] = text


EXAMPLE_QUERIES = [
    "What cases involve Section 438?",
    "Judgments discussing Article 21 and personal liberty",
    "Principles for granting anticipatory bail",
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Spectral:wght@400;600;700&family=Public+Sans:wght@400;500;600;700&display=swap');

:root {
  --ink: #16213E;
  --ink-soft: #2A3558;
  --paper: #F4F5F9;
  --card: #FFFFFF;
  --line: #DCE0EB;
  --text: #1E2433;
  --muted: #5B6478;
  --rag: #3B4BA8;
  --rag-bg: #E8EBFA;
  --kg: #0E7C7E;
  --kg-bg: #DDF3F3;
  --brass: #A97C2F;
  --brass-bg: #F6EBD3;
}

html, body, [class*="css"], .stApp {
  font-family: 'Public Sans', sans-serif;
  color: var(--text);
}
.stApp { background: var(--paper); }
#MainMenu, footer { visibility: hidden; }
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 1.2rem; padding-bottom: 4rem; padding-left: 2.5rem; padding-right: 2.5rem; max-width: 100% !important; width: 100%; }

h1, h2, h3, h4 { font-family: 'Spectral', serif; color: var(--ink); letter-spacing: -0.01em; }
h2, h3 { font-weight: 700; }

/* ---------- Hero ---------- */
.hero {
  position: relative;
  border-radius: 18px;
  padding: 2.4rem 2.6rem 2.2rem;
  color: #F3EFE4;
  background:
    repeating-linear-gradient(90deg, rgba(255,255,255,.035) 0 2px, transparent 2px 46px),
    linear-gradient(135deg, #111A33 0%, #1D2A52 100%);
  border-bottom: 4px solid var(--brass);
  overflow: hidden;
}
.hero h1 {
  font-family: 'Spectral', serif;
  font-size: 2.55rem; line-height: 1.12; font-weight: 700;
  color: #FFFFFF; margin: 0 0 .7rem 0; max-width: 1000px;
}
.hero p { font-size: 1.08rem; color: #C9D0E6; max-width: 860px; margin: 0 0 1.3rem 0; line-height: 1.55; }
.chips { display: flex; flex-wrap: wrap; gap: .5rem; }
.chip {
  font-size: .82rem; font-weight: 600; padding: .32rem .8rem; border-radius: 999px;
  border: 1px solid rgba(255,255,255,.28); color: #EDEFF8; background: rgba(255,255,255,.06);
}
.chip.kg { border-color: #4FC3C5; color: #B8F0F1; }
.chip.rag { border-color: #8E9BEA; color: #D5DAFB; }
.chip.llm { border-color: #D9B36A; color: #F3DDAE; }

.notice {
  margin: 1rem 0 1.4rem; padding: .7rem 1rem; border-radius: 10px;
  background: var(--brass-bg); border: 1px solid #E5D3A8; color: #5E4619; font-size: .88rem;
}

/* ---------- Query area ---------- */
.stTextArea textarea {
  font-family: 'Public Sans', sans-serif; font-size: 1.02rem;
  border-radius: 12px; border: 1.5px solid var(--line); background: #FFFFFF;
}
.stTextArea textarea:focus { border-color: var(--brass); box-shadow: 0 0 0 3px rgba(169,124,47,.18); }
.stTextArea label p { font-weight: 600; color: var(--ink); }
[data-testid="stForm"] {
  background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 1.2rem 1.3rem;
  box-shadow: 0 6px 24px rgba(22,33,62,.06);
}
button[kind="primaryFormSubmit"], [data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--ink); border: none; color: #fff; border-radius: 10px; font-weight: 600; padding: .65rem 1rem;
  transition: background .15s ease;
}
button[kind="primaryFormSubmit"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover { background: var(--brass); color: #fff; }
.stButton > button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
  border-radius: 999px; border: 1px solid var(--line); background: #fff; color: var(--ink-soft);
  font-size: .85rem; font-weight: 500;
}
.stButton > button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover { border-color: var(--brass); color: var(--brass); }
.hint { font-size: .85rem; color: var(--muted); margin: .9rem 0 .35rem; }

/* ---------- Section headings ---------- */
.section-title { font-family: 'Spectral', serif; font-size: 1.55rem; font-weight: 700; color: var(--ink); margin: 2.2rem 0 .2rem; }
.section-sub { color: var(--muted); font-size: .93rem; margin-bottom: 1rem; max-width: 760px; }

/* ---------- Retrieval paths ---------- */
.path {
  background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1.2rem 1.3rem 1.1rem;
  border-top-width: 5px; min-height: 210px;
}
.path.rag { border-top-color: var(--rag); }
.path.kg { border-top-color: var(--kg); }
.path-count { font-family: 'Spectral', serif; font-size: 2.6rem; font-weight: 700; line-height: 1; }
.path.rag .path-count { color: var(--rag); }
.path.kg .path-count { color: var(--kg); }
.path-label { font-size: .85rem; color: var(--muted); margin: .15rem 0 .8rem; }
.path h4 { font-size: 1.12rem; margin: 0 0 .5rem; }
.path ul { margin: 0; padding-left: 1.1rem; color: var(--text); font-size: .92rem; line-height: 1.65; }
.path .note { font-size: .82rem; color: var(--muted); margin-top: .6rem; }

.merge {
  margin-top: .9rem; background: var(--brass-bg); border: 1px solid #E5D3A8; border-radius: 14px;
  padding: 1rem 1.3rem; display: flex; flex-wrap: wrap; align-items: center; gap: 1.6rem;
}
.merge .big { font-family: 'Spectral', serif; font-size: 2.1rem; font-weight: 700; color: var(--brass); line-height: 1; }
.merge .lbl { font-size: .85rem; color: #6E5522; }
.merge .sep { width: 1px; align-self: stretch; background: #E0CD9E; }

/* ---------- Pipeline stepper ---------- */
.pipeline { display: flex; flex-wrap: wrap; gap: .6rem; margin-top: .9rem; }
.step {
  flex: 1 1 200px; background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: .85rem 1rem; position: relative;
}
.step-n {
  display: inline-flex; width: 1.6rem; height: 1.6rem; border-radius: 50%; background: var(--ink); color: #fff;
  font-size: .8rem; font-weight: 700; align-items: center; justify-content: center; margin-bottom: .45rem;
}
.step b { display: block; color: var(--ink); font-size: .95rem; margin-bottom: .15rem; }
.step span.d { font-size: .84rem; color: var(--muted); line-height: 1.45; }

/* ---------- Facts row ---------- */
.facts { display: flex; flex-wrap: wrap; gap: 1.8rem; margin: 1rem 0 .2rem; }
.fact b { font-family: 'Spectral', serif; font-size: 1.5rem; color: var(--ink); display: block; line-height: 1.1; }
.fact span { font-size: .82rem; color: var(--muted); }

/* ---------- Answer ---------- */
.st-key-answer_panel {
  background: var(--card); border: 1px solid var(--line); border-left: 6px solid var(--brass);
  border-radius: 14px; padding: 1.5rem 1.8rem; box-shadow: 0 8px 28px rgba(22,33,62,.07);
}
.st-key-answer_panel p, .st-key-answer_panel li { font-size: 1.02rem; line-height: 1.75; }
.st-key-answer_panel h1, .st-key-answer_panel h2, .st-key-answer_panel h3 { font-size: 1.25rem; margin-top: 1rem; }

/* ---------- Case cards ---------- */
.case-head { display: flex; align-items: flex-start; gap: .9rem; }
.rank {
  flex: 0 0 auto; width: 2.3rem; height: 2.3rem; border-radius: 10px; background: var(--ink); color: #fff;
  font-family: 'Spectral', serif; font-weight: 700; font-size: 1.15rem; display: flex; align-items: center; justify-content: center;
}
.case-title { flex: 1 1 auto; font-weight: 700; color: var(--ink); font-size: 1.05rem; word-break: break-word; }
.case-meta { font-weight: 400; color: var(--muted); font-size: .88rem; margin-top: .1rem; }
.badge { flex: 0 0 auto; font-size: .78rem; font-weight: 700; padding: .25rem .65rem; border-radius: 999px; white-space: nowrap; }
.badge.both { background: var(--kg-bg); color: var(--kg); border: 1px solid #A9DCDD; }
.badge.rag { background: var(--rag-bg); color: var(--rag); border: 1px solid #C3CAF0; }
.score { display: flex; align-items: center; gap: .8rem; margin: .8rem 0 .4rem; font-size: .85rem; color: var(--muted); }
.score-bar { flex: 1 1 auto; height: 8px; background: #E7EAF3; border-radius: 99px; overflow: hidden; }
.score-bar span { display: block; height: 100%; background: linear-gradient(90deg, var(--rag), var(--brass)); border-radius: 99px; }
.score b { color: var(--ink); font-variant-numeric: tabular-nums; }

.evidence { border-left: 3px solid var(--line); padding: .15rem 0 .15rem 1rem; margin: .8rem 0; }
.ev-label { font-size: .82rem; font-weight: 600; color: var(--rag); margin-bottom: .25rem; }
.evidence p { font-family: 'Spectral', serif; font-size: 1.02rem; line-height: 1.7; color: #2B3144; margin: 0; }

[data-testid="stExpander"] { border-radius: 12px; border: 1px solid var(--line); background: #fff; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
.foot { text-align: center; color: var(--muted); font-size: .82rem; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--line); }

@media (max-width: 760px) {
  .hero { padding: 1.6rem 1.3rem; }
  .hero h1 { font-size: 1.8rem; }
  .merge .sep { display: none; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# INTERACTIVE EFFECTS (cursor-reactive layer)
# ------------------------------------------------------------
FX_VERSION = "2"

FX_CSS = """
<style>
::selection { background: rgba(169,124,47,.28); }

/* hide the helper iframe that injects the effects script */
[data-testid="stElementContainer"]:has(iframe[height="0"]) {
  position: absolute; width: 0; height: 0; overflow: hidden; margin: 0; padding: 0;
}

/* ---------- Cursor: dot, trailing ring, soft glow ---------- */
.fx-dot, .fx-ring {
  position: fixed; left: 0; top: 0; pointer-events: none; opacity: 0;
}
.fx-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--brass); z-index: 2147483000; transition: opacity .25s; }
.fx-ring {
  width: 36px; height: 36px; border-radius: 50%; border: 1.5px solid rgba(169,124,47,.8); z-index: 2147482999;
  transition: opacity .25s, scale .22s cubic-bezier(.2,.8,.2,1), background-color .22s;
}
.fx-ring.hover { scale: 1.9; background-color: rgba(169,124,47,.15); }
.fx-on .fx-dot, .fx-on .fx-ring { opacity: 1; }
.fx-on .fx-dot.hide, .fx-on .fx-ring.hide { opacity: 0; }

/* ---------- Hero: torch on the pillars + knowledge-graph canvas ---------- */
.hero {
  background-image:
    radial-gradient(520px circle at var(--mx, 72%) var(--my, 30%), rgba(255,255,255,.10), transparent 62%),
    repeating-linear-gradient(90deg, rgba(255,255,255,.035) 0 2px, transparent 2px 46px),
    linear-gradient(135deg, #111A33 0%, #1D2A52 100%);
}
.hero::after {
  content: ""; position: absolute; inset: 0; pointer-events: none; z-index: 0;
  background: repeating-linear-gradient(90deg, rgba(255,255,255,.22) 0 2px, transparent 2px 46px);
  -webkit-mask-image: radial-gradient(260px circle at var(--mx, 72%) var(--my, 30%), #000, transparent 72%);
          mask-image: radial-gradient(260px circle at var(--mx, 72%) var(--my, 30%), #000, transparent 72%);
}
.fx-net { position: absolute; inset: 0; width: 100%; height: 100%; z-index: 1; pointer-events: none; }
.hero > *:not(canvas) { position: relative; z-index: 2; animation: heroIn .8s cubic-bezier(.2,.7,.2,1) both; }
.hero > h1 { animation-delay: .05s; }
.hero > p { animation-delay: .17s; }
.hero > .chips { animation-delay: .29s; }
@keyframes heroIn { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }

/* ---------- Cards: cursor spotlight + tilt ---------- */
.path, .step, .merge {
  --spot: 169,124,47;
  background-color: var(--card);
  background-image: radial-gradient(380px circle at var(--cx, -999px) var(--cy, -999px), rgba(var(--spot), .15), transparent 65%);
}
.merge { background-color: var(--brass-bg); }
.path.rag { --spot: 59,75,168; }
.path.kg { --spot: 14,124,126; }
.path, .step, .merge {
  transition: transform .14s ease-out, box-shadow .3s ease, border-color .3s ease; will-change: transform;
}
.path:hover, .step:hover { box-shadow: 0 18px 42px rgba(22,33,62,.15); }
.path.rag:hover { border-color: var(--rag); }
.path.kg:hover { border-color: var(--kg); }
[class*="st-key-case_card"] { background-color: var(--card); }

/* ---------- Buttons ---------- */
button[kind="primaryFormSubmit"], [data-testid="stBaseButton-primaryFormSubmit"] {
  position: relative; overflow: hidden; background: linear-gradient(135deg, #16213E, #34439A);
}
button[kind="primaryFormSubmit"]::after, [data-testid="stBaseButton-primaryFormSubmit"]::after {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background: linear-gradient(110deg, transparent 30%, rgba(255,255,255,.3) 50%, transparent 70%);
  transform: translateX(-120%); transition: transform .7s ease;
}
button[kind="primaryFormSubmit"]:hover::after, [data-testid="stBaseButton-primaryFormSubmit"]:hover::after { transform: translateX(120%); }
button[kind="primaryFormSubmit"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover { background: linear-gradient(135deg, #A97C2F, #C99A45); }

/* ---------- Results arriving ---------- */
.score-bar span { transform-origin: left center; animation: grow 1s cubic-bezier(.2,.7,.2,1) both; }
@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }

@media (prefers-reduced-motion: reduce) {
  .hero > *:not(canvas), .score-bar span { animation: none !important; }
}
@media (pointer: coarse) { .fx-dot, .fx-ring { display: none; } }
</style>
"""

FX_JS = r"""
(function () {
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var fine = window.matchMedia('(pointer: fine)').matches;
  var cleanups = [];
  var dead = false;
  function on(t, ev, fn, opt) { t.addEventListener(ev, fn, opt); cleanups.push(function () { t.removeEventListener(ev, fn, opt); }); }
  function make(cls) { var e = document.createElement('div'); e.className = cls; document.body.appendChild(e); cleanups.push(function () { e.remove(); }); return e; }

  var mx = -100, my = -100, rx = -100, ry = -100, moved = false;
  var dot, ring;
  if (fine && !reduce) { dot = make('fx-dot'); ring = make('fx-ring'); }

  var HOVER = 'a,button,[role="button"],summary,label,[data-baseweb="select"]';
  var TEXT = 'textarea,input';
  var CARD = '.path,.step,.merge';
  var TILT = '.path,.step,.merge';
  var lastCard = null;

  function resetCard(c) {
    if (!c) return;
    c.style.setProperty('--cx', '-999px'); c.style.setProperty('--cy', '-999px');
    if (c.matches(TILT)) c.style.transform = '';
  }

  on(document, 'mousemove', function (e) {
    mx = e.clientX; my = e.clientY;
    if (!moved) { moved = true; rx = mx; ry = my; document.body.classList.add('fx-on'); }
    var t = e.target;
    if (!t || !t.closest) return;
    if (ring) {
      var overText = !!t.closest(TEXT);
      ring.classList.toggle('hover', !!t.closest(HOVER));
      ring.classList.toggle('hide', overText);
      dot.classList.toggle('hide', overText);
    }
    var hero = t.closest('.hero');
    if (hero) {
      var r = hero.getBoundingClientRect();
      hero.style.setProperty('--mx', (e.clientX - r.left) + 'px');
      hero.style.setProperty('--my', (e.clientY - r.top) + 'px');
    }
    var card = t.closest(CARD);
    if (card !== lastCard) { resetCard(lastCard); lastCard = card; }
    if (card) {
      var b = card.getBoundingClientRect();
      var px = e.clientX - b.left, py = e.clientY - b.top;
      card.style.setProperty('--cx', px + 'px'); card.style.setProperty('--cy', py + 'px');
      if (!reduce && card.matches(TILT)) {
        var ry2 = ((px / b.width) - 0.5) * 7, rx2 = -((py / b.height) - 0.5) * 7;
        card.style.transform = 'perspective(900px) rotateX(' + rx2.toFixed(2) + 'deg) rotateY(' + ry2.toFixed(2) + 'deg) translateY(-3px)';
      }
    }
  }, { passive: true });

  on(document, 'mouseout', function (e) {
    if (!e.relatedTarget) { document.body.classList.remove('fx-on'); moved = false; resetCard(lastCard); lastCard = null; }
  });

  function tick() {
    if (dead) return;
    if (ring) {
      rx += (mx - rx) * 0.2; ry += (my - ry) * 0.2;
      var tr = 'translate3d(' + rx.toFixed(1) + 'px,' + ry.toFixed(1) + 'px,0) translate(-50%,-50%)';
      ring.style.transform = tr;
      dot.style.transform = 'translate3d(' + mx + 'px,' + my + 'px,0) translate(-50%,-50%)';
    }
    requestAnimationFrame(tick);
  }
  if (ring) requestAnimationFrame(tick);

  /* ---------- Knowledge-graph constellation in the hero ---------- */
  var LABELS = ['Sec. 438', 'Art. 21', 'Art. 14', 'Sec. 302', 'Art. 19', 'Sec. 482', 'Art. 32', 'Sec. 420'];
  var COLS = ['79,195,197', '142,155,234', '217,179,106'];

  function startNet(hero, c) {
    var ctx = c.getContext('2d');
    var W = 0, H = 0, dpr = Math.min(window.devicePixelRatio || 1, 2);
    var nodes = [], ripples = [], mouse = { x: -999, y: -999, on: false }, visible = true;

    function resize() {
      var r = hero.getBoundingClientRect();
      W = r.width; H = r.height;
      c.width = W * dpr; c.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      var n = Math.max(26, Math.round(W * H / 15000));
      nodes = [];
      for (var i = 0; i < n; i++) {
        var sp = reduce ? 0 : 0.28;
        var bx = (Math.random() - 0.5) * sp, by = (Math.random() - 0.5) * sp;
        nodes.push({
          x: Math.random() * W, y: Math.random() * H, vx: bx, vy: by, bx: bx, by: by,
          r: 1.4 + Math.random() * 1.8, c: COLS[i % 3], label: i < LABELS.length ? LABELS[i] : null
        });
      }
      if (reduce) draw();
    }

    function update() {
      for (var i = 0; i < nodes.length; i++) {
        var n = nodes[i];
        n.vx += (n.bx - n.vx) * 0.02; n.vy += (n.by - n.vy) * 0.02;
        if (mouse.on) {
          var dx = mouse.x - n.x, dy = mouse.y - n.y, d = Math.hypot(dx, dy);
          if (d < 200 && d > 1) { n.vx += dx / d * 0.02 * (1 - d / 200); n.vy += dy / d * 0.02 * (1 - d / 200); }
        }
        n.x += n.vx; n.y += n.vy;
        if (n.x < 0 || n.x > W) { n.vx *= -1; n.bx *= -1; n.x = Math.max(0, Math.min(W, n.x)); }
        if (n.y < 0 || n.y > H) { n.vy *= -1; n.by *= -1; n.y = Math.max(0, Math.min(H, n.y)); }
      }
      for (var k = ripples.length - 1; k >= 0; k--) { ripples[k].t += 0.018; if (ripples[k].t >= 1) ripples.splice(k, 1); }
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      var i, j, a, b, d, al;
      ctx.lineWidth = 1;
      for (i = 0; i < nodes.length; i++) {
        a = nodes[i];
        for (j = i + 1; j < nodes.length; j++) {
          b = nodes[j];
          d = Math.hypot(a.x - b.x, a.y - b.y);
          if (d < 130) {
            al = (1 - d / 130) * 0.26;
            ctx.strokeStyle = 'rgba(190,200,235,' + al.toFixed(3) + ')';
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          }
        }
      }
      if (mouse.on) {
        ctx.lineWidth = 1.2;
        for (i = 0; i < nodes.length; i++) {
          a = nodes[i];
          d = Math.hypot(a.x - mouse.x, a.y - mouse.y);
          if (d < 190) {
            al = (1 - d / 190) * 0.85;
            ctx.strokeStyle = 'rgba(217,179,106,' + al.toFixed(3) + ')';
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(mouse.x, mouse.y); ctx.stroke();
          }
        }
        ctx.fillStyle = 'rgba(217,179,106,.95)';
        ctx.beginPath(); ctx.arc(mouse.x, mouse.y, 3, 0, Math.PI * 2); ctx.fill();
      }
      ctx.font = '11px "Public Sans", sans-serif';
      for (i = 0; i < nodes.length; i++) {
        a = nodes[i];
        d = mouse.on ? Math.hypot(a.x - mouse.x, a.y - mouse.y) : 999;
        var near = d < 110;
        ctx.fillStyle = 'rgba(' + a.c + ',' + (near ? 1 : 0.8) + ')';
        ctx.beginPath(); ctx.arc(a.x, a.y, a.r + (near ? 1.6 : 0), 0, Math.PI * 2); ctx.fill();
        if (a.label) {
          ctx.fillStyle = 'rgba(232,235,246,' + (near ? 0.97 : 0.5) + ')';
          ctx.fillText(a.label, a.x + 8, a.y + 3);
        }
      }
      for (i = 0; i < ripples.length; i++) {
        var rp = ripples[i];
        ctx.strokeStyle = 'rgba(217,179,106,' + ((1 - rp.t) * 0.55).toFixed(3) + ')';
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(rp.x, rp.y, rp.t * 280, 0, Math.PI * 2); ctx.stroke();
      }
    }

    var ro = new ResizeObserver(resize); ro.observe(hero);
    var io = new IntersectionObserver(function (en) { visible = en[0].isIntersecting; }); io.observe(hero);

    function pos(e) {
      var r = hero.getBoundingClientRect();
      mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top; mouse.on = true;
    }
    if (!reduce) {
      hero.addEventListener('mousemove', pos);
      hero.addEventListener('mouseleave', function () { mouse.on = false; });
      hero.addEventListener('click', function (e) {
        pos(e);
        ripples.push({ x: mouse.x, y: mouse.y, t: 0 });
        nodes.forEach(function (n) {
          var dx = n.x - mouse.x, dy = n.y - mouse.y, d = Math.hypot(dx, dy) || 1;
          if (d < 240) { var f = (1 - d / 240) * 3.4; n.vx += dx / d * f; n.vy += dy / d * f; }
        });
      });
    }

    function frame() {
      if (dead || !c.isConnected) { ro.disconnect(); io.disconnect(); return; }
      if (visible && !document.hidden) { update(); draw(); }
      requestAnimationFrame(frame);
    }
    resize();
    if (!reduce) requestAnimationFrame(frame);
  }

  function ensureCanvas() {
    var hero = document.querySelector('.hero');
    if (!hero || hero.querySelector('canvas.fx-net')) return;
    var c = document.createElement('canvas');
    c.className = 'fx-net';
    hero.insertBefore(c, hero.firstChild);
    startNet(hero, c);
  }

  /* ---------- Count-up for result numbers ---------- */
  function countUps() {
    var items = document.querySelectorAll('.cu:not([data-done])');
    items.forEach(function (e) {
      e.setAttribute('data-done', '1');
      var raw = e.textContent.trim();
      var target = parseInt(raw.replace(/,/g, ''), 10);
      if (isNaN(target) || reduce) return;
      var t0 = performance.now(), dur = 900;
      (function step(t) {
        var p = Math.min(1, (t - t0) / dur);
        var v = Math.round(target * (1 - Math.pow(1 - p, 3)));
        e.textContent = v.toLocaleString();
        if (p < 1) requestAnimationFrame(step); else e.textContent = raw;
      })(t0);
    });
  }

  var queued = false;
  var mo = new MutationObserver(function () {
    if (queued) return;
    queued = true;
    requestAnimationFrame(function () { queued = false; ensureCanvas(); countUps(); });
  });
  mo.observe(document.body, { childList: true, subtree: true });
  ensureCanvas(); countUps();

  window.__legalFxCleanup = function () {
    dead = true;
    mo.disconnect();
    cleanups.forEach(function (f) { f(); });
    document.querySelectorAll('canvas.fx-net').forEach(function (c) { c.remove(); });
  };
})();
"""

st.markdown(FX_CSS, unsafe_allow_html=True)
components.html(
    "<script>(function(){var P=window.parent,D=P.document;"
    "var old=D.getElementById('legal-fx');"
    "if(old){if(old.getAttribute('data-v')==='%s')return;"
    "if(P.__legalFxCleanup)P.__legalFxCleanup();old.remove();}"
    "var s=D.createElement('script');s.id='legal-fx';s.setAttribute('data-v','%s');"
    "s.textContent=%s;D.head.appendChild(s);})();</script>"
    % (FX_VERSION, FX_VERSION, json.dumps(FX_JS)),
    height=0,
)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
render_html("""
<div class="hero">
  <h1>Search Indian judgments with answers you can trace to the source</h1>
  <p>Ask a legal question. The system looks through judgments using keyword and semantic search together with a
  Knowledge Graph of sections and articles, then writes an answer from the passages it found.</p>
  <div class="chips">
    <span class="chip rag">BM25 keyword search</span>
    <span class="chip rag">SBERT + FAISS semantic search</span>
    <span class="chip kg">Knowledge Graph</span>
    <span class="chip llm">RAG-based answer</span>
  </div>
</div>
<div class="notice">
  For legal research and education only. This prototype is not professional legal advice and does not replace
  a qualified legal professional or a judicial decision.
</div>
""")

# ------------------------------------------------------------
# QUERY INPUT
# ------------------------------------------------------------
render_html('<div class="hint">Try an example</div>')
example_cols = st.columns(len(EXAMPLE_QUERIES))
for col, example in zip(example_cols, EXAMPLE_QUERIES):
    col.button(example, key=f"ex_{example}", on_click=set_example_query, args=(example,), use_container_width=True)

with st.form("legal_query_form", clear_on_submit=False):
    question = st.text_area(
        "Your legal question",
        key="legal_query",
        placeholder="Example: What cases involve Section 438?",
        height=105,
    )
    search_button = st.form_submit_button("Search judgments", type="primary", use_container_width=True)

# ------------------------------------------------------------
# SEARCH + RESULTS
# ------------------------------------------------------------
if search_button:
    if not question.strip():
        st.error("Enter a legal question to start the search.")
        st.stop()

    query = question.strip()

    with st.spinner("Loading the indexed repository and Knowledge Graph..."):
        try:
            bm25, bm25_metadata, faiss_index, faiss_metadata, embedding_model = load_rag_system()
            graph, node_lookup = load_knowledge_graph()
        except Exception as exc:
            st.error(f"Could not load retrieval components: {type(exc).__name__}: {exc}")
            st.stop()

    from authority_rerank import search as rag_search
    from context_builder import build_context_package, create_llm_context

    with st.spinner("Running both retrieval paths and combining their candidates..."):
        try:
            # Existing hybrid RAG path: BM25 + FAISS + RRF.
            rag_original_cases, _, _ = rag_search(
                query, bm25, bm25_metadata, faiss_index, faiss_metadata, embedding_model
            )
            rag_case_ids = {
                case.get("case_id") for case in rag_original_cases if case.get("case_id")
            }

            # Separate KG path: explicit Section/Article matching and graph traversal.
            query_entities = extract_structured_query_entities(query)
            matched_nodes = find_kg_nodes(graph, query_entities)
            kg_case_ids = get_kg_case_candidates(graph, node_lookup, matched_nodes)
            st.session_state.kg_case_ids = kg_case_ids

            # Union and deduplicate candidates from both paths.
            combined_ids = rag_case_ids | kg_case_ids
            if not combined_ids:
                st.warning("Neither retrieval path found a candidate case. Try rewording the question or naming a section or article.")
                st.stop()

            _, filtered_cases, final_cases = make_candidate_specific_results(
                query, bm25, bm25_metadata, faiss_index, faiss_metadata,
                embedding_model, combined_ids
            )
            final_cases = final_cases[:5]
            for case in final_cases:
                case["kg_match"] = case.get("case_id") in kg_case_ids

            context_package = build_context_package(query, final_cases)
            legal_context = create_llm_context(context_package)
        except Exception as exc:
            st.error(f"Combined KG + RAG retrieval failed: {type(exc).__name__}: {exc}")
            st.exception(exc)
            st.stop()

    overlap = len(rag_case_ids & kg_case_ids)

    # ---------------- Answer first ----------------
    render_html(f"""
    <div class="section-title">Answer</div>
    <div class="section-sub">Written from the top {len(final_cases)} ranked judgments. Check the cited cases below before relying on it.</div>
    """)

    with st.spinner("Writing a RAG-based answer from the retrieved passages..."):
        try:
            from generate_rag_answer import generate_answer
            answer = generate_answer(query, legal_context)
        except Exception as exc:
            st.error(f"Answer generation failed: {type(exc).__name__}: {exc}")
            st.stop()

    with st.container(key="answer_panel"):
        st.markdown(answer)

    # ---------------- Ranked cases ----------------
    render_html(f"""
    <div class="section-title">Ranked cases and evidence</div>
    <div class="section-sub">A Knowledge Graph match means the case is linked to the section or article you named. It does not
    show the case is correct or controlling authority.</div>
    """)

    max_score = max(
        [float(c.get("final_score", 0.0) or 0.0) for c in final_cases] + [1e-9]
    )
    for rank, case in enumerate(final_cases, start=1):
        is_kg = bool(case.get("kg_match"))
        badge = '<span class="badge both">Knowledge Graph + RAG</span>' if is_kg else '<span class="badge rag">RAG</span>'
        score = float(case.get("final_score", 0.0) or 0.0)
        pct = max(4.0, min(100.0, score / max_score * 100.0))
        with st.container(border=True, key=f"case_card_{rank}"):
            render_html(f"""
            <div class="case-head">
              <div class="rank">{rank}</div>
              <div class="case-title">{esc(case.get('case_id', 'Unknown case'))}
                <div class="case-meta">{esc(case.get('court', 'Court not recorded'))}</div>
              </div>
              {badge}
            </div>
            <div class="score">
              <span>Relevance</span>
              <div class="score-bar"><span style="width:{pct:.0f}%"></span></div>
              <b>{score:.4f}</b>
            </div>
            """)
            chunks = case.get("supporting_chunks", [])[:3]
            with st.expander(f"Supporting passages ({len(chunks)})", expanded=(rank == 1)):
                if not chunks:
                    st.caption("No supporting passages were returned for this case.")
                for i, chunk in enumerate(chunks, start=1):
                    pages = chunk.get("pages", []) or [chunk.get("start_page", "?")]
                    page_text = ", ".join(str(p) for p in pages)
                    body = esc(chunk.get("text", "No text available")).replace("\n", "<br>")
                    render_html(f"""
                    <div class="evidence">
                      <div class="ev-label">Passage {i}, page {esc(page_text)}</div>
                      <p>{body}</p>
                    </div>
                    """)

    # ---------------- How the search worked ----------------
    render_html("""
    <div class="section-title">How the search worked</div>
    <div class="section-sub">Two retrieval paths ran side by side, then their cases were merged and ranked together.</div>
    """)

    left, right = st.columns(2, gap="large")
    with left:
        render_html(f"""
        <div class="path rag">
          <div class="path-count cu">{len(rag_case_ids)}</div>
          <div class="path-label">cases from hybrid retrieval</div>
          <h4>Keyword + semantic search</h4>
          <ul>
            <li>BM25 matches the exact words in your question</li>
            <li>SBERT + FAISS matches by meaning</li>
            <li>Reciprocal Rank Fusion (RRF) merges the two rankings</li>
          </ul>
        </div>
        """)
        with st.expander("Hybrid retrieval candidate cases"):
            st.write(sorted(rag_case_ids) if rag_case_ids else "No hybrid retrieval candidates found.")
    with right:
        render_html(f"""
        <div class="path kg">
          <div class="path-count cu">{len(kg_case_ids)}</div>
          <div class="path-label">cases from the Knowledge Graph</div>
          <h4>Section and article lookup</h4>
          <ul>
            <li>Finds the Section or Article named in your question</li>
            <li>Follows graph links to cases that involve it</li>
          </ul>
          <div class="note">This path only activates when the question names a section or article.</div>
        </div>
        """)
        with st.expander("Knowledge Graph matching details"):
            st.write("Detected references:", query_entities or "No explicit Section/Article reference detected")
            if matched_nodes:
                st.dataframe([
                    {
                        "Node type": n.get("node_type", n.get("type", "")),
                        "Node ID": n.get("node_id", ""),
                        "Name": (n.get("properties", {}) or {}).get(
                            "section_name", (n.get("properties", {}) or {}).get("article_name", "")
                        ),
                    }
                    for n in matched_nodes
                ], use_container_width=True, hide_index=True)
            else:
                st.info("No Section or Article node matched. Hybrid retrieval still finds cases for general questions.")
            st.write("Knowledge Graph case candidates:")
            st.write(sorted(kg_case_ids) if kg_case_ids else "No Knowledge Graph case candidates for this query.")

    render_html(f"""
    <div class="merge">
      <div><div class="big cu">{len(combined_ids)}</div><div class="lbl">unique cases after merging</div></div>
      <div class="sep"></div>
      <div><div class="big cu">{overlap}</div><div class="lbl">found by both paths</div></div>
      <div class="sep"></div>
      <div><div class="big cu">{len(context_package.get('evidence_records', []))}</div><div class="lbl">evidence records collected</div></div>
    </div>

    <div class="pipeline">
      <div class="step"><div class="step-n">1</div><b>Merge candidates</b>
        <span class="d">Combine both paths and remove duplicate cases.</span></div>
      <div class="step"><div class="step-n">2</div><b>Rank by relevance and authority</b>
        <span class="d">Re-score passages, group them by case, drop weak matches, and rerank.</span></div>
      <div class="step"><div class="step-n">3</div><b>Build the context</b>
        <span class="d">Take the top five cases with their supporting passages and metadata.</span></div>
      <div class="step"><div class="step-n">4</div><b>Write the answer</b>
        <span class="d">The RAG answer is written only from the packaged passages.</span></div>
    </div>

    <div class="facts">
      <div class="fact"><b class="cu">{context_package.get('cases_considered', 0)}</b><span>cases sent to answer generation</span></div>
      <div class="fact"><b class="cu">{context_package.get('total_context_characters', 0):,}</b><span>characters of context</span></div>
    </div>
    """)

    render_html("""
    <div class="foot">For legal research and education only. Verify cited judgments and applicable law with
    authoritative sources and qualified legal professionals.</div>
    """)

ARCHITECTURE_IMAGE = PROJECT_ROOT / "System Architecture.png"

render_html("""
<div class="section-title">System architecture</div>
<div class="section-sub">How a legal query moves through hybrid retrieval, Knowledge Graph retrieval, ranking and answer generation.</div>
""")

with st.expander("View system architecture diagram", expanded=False):
    if ARCHITECTURE_IMAGE.exists():
        _left, _center, _right = st.columns([1, 2, 1])
        with _center:
            st.image(
                str(ARCHITECTURE_IMAGE),
                caption="End-to-end KG-enhanced RAG pipeline for Indian legal case retrieval",
                use_container_width=True,
            )
    else:
        st.warning(f"Architecture image not found at: {ARCHITECTURE_IMAGE}")
        st.code("""
Legal Query
    |
Query Processing
    |-----------------------------|
    v                             v
Hybrid Retrieval             KG Retrieval
BM25 + FAISS + RRF           Section/Article matching
    |                        + graph traversal
    |                             |
    |-------------|---------------|
                  v
       Candidate Union / Deduplication
                  |
       Case-Level Processing
                  |
       Relevance & Authority-Aware Ranking
                  |
       Context Construction (Top 5 cases)
                  |
       RAG Answer Generation
                  |
       Legal Answer + Supporting Cases/Evidence
""", language="text")