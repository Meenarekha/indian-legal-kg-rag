# Indian Legal Case Retrieval & Question-Answering System

### KG-Enhanced Retrieval-Augmented Generation (RAG) for Indian Legal Research

An intelligent legal research prototype that retrieves relevant Indian judicial judgments using **hybrid keyword and semantic search, Knowledge Graph retrieval, and evidence-grounded answer generation** with Google Gemini.

## 🚀 Live Demo

**Streamlit App:** [Open the application](https://indian-legal-kg-rag.streamlit.app/)

> Replace the URL above with your actual Streamlit Community Cloud application URL.

**Source Code:** [GitHub Repository](https://github.com/Meenarekha/indian-legal-kg-rag)

**DVC Data Storage:** [DagsHub Repository](https://dagshub.com/Meena/indian-legal-kg-rag)

## 🎥 Project Demo

[▶️ Watch the Project Demo](./Project%20demo.mp4)

The demonstration covers:
- Legal query input through the Streamlit interface.
- Hybrid retrieval using BM25 and FAISS.
- Knowledge Graph-based case retrieval.
- Case ranking and evidence selection.
- Gemini-generated answers with supporting legal evidence.

## 📌 Overview

Finding relevant judicial precedents requires searching large collections of legal judgments and examining lengthy documents. This project simplifies legal research by retrieving relevant cases, ranking supporting evidence, and generating answers grounded in retrieved judicial passages.

The system combines two retrieval paths:

- **Hybrid RAG Retrieval:** BM25 keyword search and SBERT embeddings with FAISS semantic search. Reciprocal Rank Fusion (RRF) combines their rankings.
- **Knowledge Graph Retrieval:** Matches statutory Section and constitutional Article references and traverses graph relationships to identify related cases.

Candidates from both paths are combined and deduplicated before case-level processing, relevance filtering, authority-aware reranking, and context construction. Google Gemini generates answers from the retrieved judicial evidence.

## ✨ Key Features

- PDF text extraction using PyMuPDF.
- Legal-aware text cleaning and chunking.
- BM25 keyword-based retrieval.
- SBERT embeddings and FAISS vector search.
- Hybrid retrieval using Reciprocal Rank Fusion.
- Case-level aggregation and authority-aware reranking.
- Knowledge Graph retrieval through Section/Article matching.
- Candidate combination across Knowledge Graph and hybrid RAG.
- Evidence-grounded answer generation using Google Gemini.
- Supporting cases and judicial passages displayed in the Streamlit interface.
- Evaluation using a fixed set of 30 legal research questions.

## 🏗️ System Architecture

```text
Indian Court Judgments (PDF)
             |
             v
       Preprocessing
             |
             v
      Indexing & Storage
        /           \
       v             v
   BM25 Index    FAISS Vector Store
        \           /
         v         v
      Hybrid Retrieval
       BM25 + FAISS
             |
          RRF Fusion

Legal Query
     |
     v
Query Processing
     |
     +-------------------+
     |                   |
     v                   v
Hybrid Retrieval    Knowledge Graph
     |                   |
     +---------+---------+
               |
               v
        Candidate Union
      (Combine and Deduplicate)
               |
               v
      Case-Level Processing
               |
               v
     Relevance & Authority-
         Aware Ranking
               |
               v
       Context Construction
               |
               v
        Google Gemini API
               |
               v
      Legal Answer + Evidence
```

The Knowledge Graph is a separate retrieval resource. It complements, rather than replaces, the BM25 and FAISS indexes.

## 🗂️ Data and DVC Management

The project uses **DVC (Data Version Control)** with DagsHub to manage large retrieval resources separately from the GitHub source code.

- **GitHub:** Stores application code, configuration, and DVC pointer files.
- **DagsHub DVC storage:** Stores large retrieval indexes and processed data.
- **Streamlit Community Cloud:** Hosts the web application.

The repository tracks DVC metadata for:

```text
data/
├── indexes.dvc
├── processed.dvc
├── indexes/
└── processed/
```

The actual contents of `indexes/` and `processed/` are retrieved through DVC. This avoids rebuilding the retrieval indexes whenever the application is deployed, provided the required DVC data is available.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Web interface | Streamlit |
| PDF extraction | PyMuPDF |
| Keyword retrieval | BM25 |
| Embeddings | Sentence Transformers (SBERT) |
| Vector search | FAISS |
| Ranking fusion | Reciprocal Rank Fusion (RRF) |
| Structured retrieval | Knowledge Graph |
| Answer generation | Google Gemini API |
| Data versioning | DVC |
| Remote data storage | DagsHub |
| Source control | Git and GitHub |
| Deployment | Streamlit Community Cloud |

## 📁 Project Structure

```text
indian-legal-kg-rag/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .dvc/
│   └── config
├── data/
│   ├── indexes.dvc
│   ├── processed.dvc
│   ├── indexes/
│   ├── processed/
│   └── evaluation/
├── knowledge_graph/
├── scripts/
└── docs/
```

*The structure above is illustrative; some directories may differ in the actual repository.*

## ⚙️ Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/Meenarekha/indian-legal-kg-rag.git
cd indian-legal-kg-rag
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

Ensure `requirements.txt` includes the packages required by the application, including Streamlit and DVC with S3 support.

```text
streamlit
dvc[s3]
```

Retain all other dependencies used by your project.

### 4. Configure credentials

Configure the DagsHub DVC remote and credentials according to the repository's DagsHub setup instructions.

For local development, keep secrets in local configuration or environment variables. Do not commit credentials or API keys to GitHub.

Configure the Gemini API key using an environment variable or the secrets mechanism supported by the application:

```text
GEMINI_API_KEY=your_gemini_api_key
```

### 5. Download the indexes and processed data

```powershell
dvc pull -r origin
```

This downloads the data referenced by the tracked DVC pointer files. The DagsHub credentials must have permission to access the remote storage.

### 6. Run the application

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal.

## ☁️ Deployment on Streamlit Community Cloud

The application can be deployed by connecting the GitHub repository to Streamlit Community Cloud.

### Deployment workflow

1. Push the application code and DVC pointer files to GitHub.
2. Ensure the DagsHub DVC remote configuration is available in the repository.
3. Add the required DagsHub credentials and Gemini API key in the Streamlit application's **Secrets** settings.
4. Install the project dependencies from `requirements.txt`.
5. Execute `dvc pull -r origin` before loading the BM25 and FAISS indexes.
6. Launch `app.py` and test the retrieval pipeline.

The application must download its required DVC data before initializing the retrieval components. If DVC authentication or file retrieval fails, deployment should stop with a useful error message rather than attempting to load missing indexes.

**Important:** Never store access tokens, secret keys, or API keys in source code, GitHub commits, or public configuration files. Use Streamlit Secrets or another suitable secret-management mechanism.

## 💡 Example Queries

Try the following questions:

1. What cases involve Section 438?
2. What judgments discuss Article 21 and personal liberty?
3. What are the principles of natural justice?
4. What factors are considered when granting anticipatory bail?
5. How do courts interpret personal liberty?

Queries containing explicit Section or Article references can use the structured Knowledge Graph retrieval path. General legal questions primarily depend on hybrid RAG retrieval.

## 📊 Evaluation

The project uses a fixed set of 30 legal research questions and reference answers.

The evaluation process examines:

- Retrieval relevance and ranking.
- Consistency with reference answers.
- Answer completeness.
- Support for generated claims in retrieved judicial evidence.

Potential retrieval metrics include Recall@K, Precision@K, MRR, and nDCG. Automated evaluation helps compare system configurations but does not establish professional legal correctness.

## ⚠️ Limitations

- The current corpus contains a limited collection of Indian judicial judgments and does not represent every Indian court decision.
- Knowledge Graph retrieval primarily supports explicit Section and Article references.
- General legal concepts may not always produce Knowledge Graph candidates.
- Similar section numbers can occur across different statutes.
- Retrieval and generated answers may be incomplete or incorrect.
- Automated evaluation does not replace legal expert review.

## 🔮 Future Enhancements

### 1. Indian Kanoon API Integration
Integrate an external legal-document retrieval source, subject to API access, usage limits, authentication, and licensing terms.

### 2. Expanded Knowledge Graph
Add legal issue entities and improve retrieval for concepts such as personal liberty, natural justice, and judicial review.

### 3. Retrieval Evaluation
Compare baseline hybrid RAG against Knowledge Graph-enhanced RAG using the same benchmark queries and retrieval metrics.

### 4. Evidence and Citation Verification
Improve source traceability, citation consistency, and detection of claims unsupported by retrieved passages.

### 5. User Feedback
Collect user feedback to improve usability and identify recurring retrieval problems.

## ⚖️ Legal Disclaimer

This project is intended for **legal research and educational purposes only**. It does not provide professional legal advice and does not replace a qualified legal professional or a judicial decision.

Users should independently verify judgments, citations, applicable legislation, and subsequent legal developments using reliable legal sources.

## 🙏 Acknowledgements

This project uses open-source libraries for document processing, information retrieval, embeddings, vector search, and web application development. Google Gemini is used for answer generation, while DVC and DagsHub support retrieval-data management and storage.
