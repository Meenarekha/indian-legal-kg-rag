# Indian Legal Case Retrieval & Question-Answering System

### KG-Enhanced Retrieval-Augmented Generation (RAG) for Indian Legal Research

An intelligent legal research prototype that retrieves relevant Indian judicial judgments using **hybrid keyword and semantic search**, **Knowledge Graph retrieval**, and **evidence-grounded answer generation** with Google Gemini.

## Demo Video

Watch the screen-recorded demonstration of the Indian Legal Case Retrieval & Question-Answering System.

[▶️ **Watch the Project Demo**](C:\Legal Query RAG\Project demo.mp4)

The demonstration covers:
- Legal query input through the Streamlit interface.
- Hybrid retrieval using BM25 and FAISS.
- Knowledge Graph-based case retrieval.
- Case ranking and evidence selection.
- Gemini-generated answers with supporting legal evidence.

## Overview

Finding relevant judicial precedents can require searching large collections of legal judgments and examining lengthy documents. This project aims to simplify legal research by retrieving relevant cases, ranking supporting evidence, and generating answers grounded in retrieved judicial passages.

The system combines two retrieval paths:

- **Hybrid RAG Retrieval:** Uses BM25 keyword retrieval and SBERT embeddings with FAISS semantic retrieval. Reciprocal Rank Fusion (RRF) combines the rankings.
- **Knowledge Graph Retrieval:** Matches explicit statutory Section and constitutional Article references and traverses graph relationships to identify related case candidates.

The candidates from both paths are combined and deduplicated before case-level processing, relevance filtering, authority-aware reranking, and context construction. Google Gemini then generates an answer using the retrieved judicial evidence.

## Key Features

- PDF text extraction using PyMuPDF.
- Text cleaning and legal-aware chunking.
- BM25 keyword-based retrieval.
- SBERT-based dense embeddings and FAISS vector search.
- Hybrid retrieval using Reciprocal Rank Fusion.
- Case-level aggregation and authority-aware reranking.
- Knowledge Graph retrieval through Section/Article matching and graph traversal.
- Candidate combination across KG and hybrid RAG.
- Evidence-based answer generation using Google Gemini.
- Supporting cases and judicial passages displayed in the Streamlit UI.
- Retrieval and answer evaluation using a fixed set of reference questions.

## System Architecture

```text
Indian Court Judgments (PDF)
             |
             v
      Preprocessing
             |
             v
       Indexing & Storage
       /               \
   BM25 Index       FAISS Vector Store
       \               /
        Hybrid Retrieval
        BM25 + FAISS + RRF
                  \
                   \
Legal Query ---> Query Processing
                   /          \
                  v            v
          Hybrid Retrieval   KG Retrieval
          BM25 + FAISS       Section/Article
          + RRF              Matching
                  \            /
                   \          /
                    v        v
               Candidate Union
             (Combine and Deduplicate)
                       |
                       v
              Case-Level Processing
                       |
                       v
          Relevance and Authority-Aware
                    Ranking
                       |
                       v
              Context Construction
                       |
                       v
            Google Gemini API
                       |
                       v
          Legal Answer + Supporting
               Cases and Evidence
```

The Knowledge Graph is a separate retrieval resource. It does not replace the BM25 or FAISS indexes; its candidate cases are combined with those returned by hybrid retrieval.

## Dataset

The current local development corpus contains **205 Indian court judgment PDFs**:

| Court category | Number of PDFs |
|---|---:|
| District Court | 60 |
| High Court | 40 |
| Supreme Court | 105 |
| **Total** | **205** |

The judgments are processed into searchable chunks while preserving source metadata, including case identifiers, court information, and page references.

The raw PDF collection and large generated retrieval indexes are not included in the repository by default. Users must have access to the required source documents and prepare the local retrieval resources before running the complete application.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| User interface | Streamlit |
| PDF extraction | PyMuPDF |
| Keyword retrieval | BM25 |
| Embeddings | Sentence Transformers (SBERT) |
| Vector search | FAISS |
| Ranking fusion | Reciprocal Rank Fusion (RRF) |
| Structured retrieval | Knowledge Graph |
| Answer generation | Google Gemini API |
| Evaluation | Automated reference-consistency and evidence-support evaluation |

## Project Structure

The following is a conceptual overview. The exact paths may vary depending on the files included in the repository.

```text
Legal-Query-RAG/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── scripts/
│   ├── preprocessing
│   ├── hybrid retrieval
│   ├── ranking
│   ├── context construction
│   ├── answer generation
│   └── evaluation
├── knowledge_graph/
│   ├── README.md
│   ├── scripts/
│   └── data/
│       ├── nodes/
│       └── relationships/
├── data/
│   └── evaluation/
└── docs/
    ├── architecture.png
    └── screenshots/
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/indian-legal-kg-rag.git
cd indian-legal-kg-rag
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Copy `.env.example` to `.env` and add your own Gemini API key:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
```

Never commit the real `.env` file or expose API keys in source code.

### 5. Prepare the local data and retrieval resources

Place the judgment PDFs and required generated retrieval resources in the paths expected by the project scripts. Run the relevant preprocessing, indexing, and Knowledge Graph construction scripts if these resources have not already been generated.

The application depends on these local resources; cloning the source code alone does not automatically download the corpus or build the indexes.

### 6. Launch the application

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit in your browser.

## Example Queries

Try the following questions in the application:

1. What cases involve Section 438?
2. What cases discuss Article 21?
3. What are the principles of natural justice?
4. What factors are considered when granting anticipatory bail?
5. How do courts interpret personal liberty?

Queries containing explicit Section or Article references can use the current structured KG retrieval path. General legal questions primarily depend on hybrid RAG retrieval.

## Evaluation

The project uses a fixed set of 30 legal research questions and reference answers for evaluation.

The evaluation process examines aspects such as:

- Retrieval relevance and ranking.
- Consistency with reference answers.
- Answer completeness.
- Support for generated claims in retrieved judicial evidence.

Automated evaluation is useful for comparing system outputs, but it does not establish professional legal correctness. Groundedness indicates whether an answer is supported by retrieved evidence; it is not a substitute for legal expert validation.

A future experimental comparison should evaluate baseline hybrid RAG against KG-enhanced RAG using the same queries and retrieval metrics.

## Limitations

- The local corpus is limited to the collected 205 judgments and does not represent every Indian judicial decision.
- Current KG retrieval primarily supports explicit Section and Article references.
- General legal concepts may not always produce Knowledge Graph candidates.
- Similar section numbers can occur across different statutes; retrieved evidence must be checked in its legal context.
- Automated evaluation does not replace legal expert review.
- Retrieval and generated answers may be incomplete or incorrect.

This project is a research prototype, not a substitute for professional legal advice.

## Future Enhancements

### 1. Indian Kanoon API Integration

Integrate the Indian Kanoon API as an additional source of legal-document retrieval.

Planned capabilities:

- Search for judgments beyond the existing local corpus.
- Retrieve additional case metadata and document content where API access permits.
- Merge external results with local KG and hybrid RAG candidates.
- Deduplicate cases and preserve document identifiers, source URLs, and provenance.
- Use relevant external passages as additional evidence for answer generation.
- Provide clear source attribution and handle API errors or usage limits.

**Status:** Planned future enhancement. API integration is not part of the current implementation.

Before implementation, verify API access, authentication, pricing, rate limits, permitted usage, and document redistribution terms. The initial enhancement should retrieve relevant documents on demand rather than attempt to download the entire legal database.

### 2. Expanded Knowledge Graph

- Add LegalIssue entities and connect them to relevant cases.
- Improve retrieval for concepts such as natural justice, personal liberty, and judicial review.
- Add validated relationships between cited cases where reliable evidence is available.

### 3. Retrieval Evaluation

- Compare baseline RAG and KG-enhanced RAG.
- Measure Recall@K, Precision@K, MRR, nDCG, and retrieval latency.
- Evaluate improvements using the same benchmark queries.

### 4. Evidence and Citation Verification

- Improve source traceability and citation consistency.
- Detect claims that are not supported by retrieved passages.
- Investigate automated cross-checking methods while clearly reporting their limitations.

### 5. User Satisfaction Feedback

- Allow users to indicate whether an answer was helpful.
- Store feedback for analysis and identify recurring usability problems.
- Treat satisfaction ratings as user experience feedback, not legal correctness validation.

## Legal Disclaimer

This system is intended for **legal research and educational assistance only**. It does not provide professional legal advice, does not replace a qualified legal professional, and must not be treated as a judicial decision or authoritative determination of legal rights.

Users should verify relevant judgments, citations, applicable legislation, and subsequent legal developments using reliable legal sources.

## Acknowledgements

This project uses open-source software libraries for document processing, information retrieval, embeddings, vector search, and application development. Google Gemini is used for answer generation. Additional external legal-data providers may be integrated in future versions, subject to their access and usage terms.
