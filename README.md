# 🤖 AI Recruiter Ranking System

An AI-powered candidate ranking system designed to intelligently match candidates with job descriptions using Semantic AI Retrieval, Vector Embeddings, and Hybrid Scoring techniques.

---

# 🚀 Problem Statement

Traditional recruitment systems rely heavily on keyword-based filtering, often missing highly relevant candidates. This project builds an intelligent AI recruiter system that understands contextual relevance, behavioral signals, and candidate quality instead of relying only on keyword matching.

---

# 🎯 Project Objective

The objective of this system is to:

- Understand job descriptions semantically
- Evaluate candidates beyond keyword overlap
- Rank candidates using AI-powered hybrid scoring
- Generate recruiter-ready Top-100 candidate shortlists

---

# 🧠 Core Features

✅ Semantic Candidate Retrieval  
✅ Embedding-Based Matching  
✅ Hybrid Scoring System  
✅ Behavioral Signal Analysis  
✅ Recruiter-Ready Candidate Ranking  
✅ Streamlit Interactive Dashboard  
✅ CSV Export Support  
✅ Top-100 Candidate Generation  
✅ Large-Scale Candidate Processing (~100K Profiles)

---

# ⚙️ Technologies Used

- Python
- Sentence Transformers
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- GitHub

---

# 🏗️ System Architecture

The workflow of the system:

1. Job Description Input
2. JD Preprocessing & Skill Extraction
3. Embedding Generation
4. Semantic Retrieval
5. Behavioral Signal Analysis
6. Hybrid Scoring
7. Candidate Ranking
8. Top-100 Output Generation

---

# 📊 Ranking Methodology

The system ranks candidates using:

- Semantic similarity between JD and candidate profiles
- Cosine similarity scoring
- Recruiter engagement signals
- GitHub activity
- Profile completeness
- Interview completion rate
- Hybrid weighted ranking strategy

---

# 🛡️ Explainability & Validation

The system provides:

- Transparent ranking explanations
- Structured scoring pipeline
- Suspicion penalty handling
- Reliable candidate evaluation
- Validated CSV submission generation

---

# 🖥️ Streamlit Dashboard

The project includes a Streamlit-based recruiter dashboard for:

- Job Description input
- Candidate ranking
- Score visualization
- CSV download support

---

# 📂 Project Structure

```bash
AI-Recruiter-System/
│
├── app.py
├── fast_ranker.py
├── requirements.txt
├── sample_candidates.json
├── submission.csv
├── architecture.png
├── presentation.pptx
└── README.md
