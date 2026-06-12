import pickle
import numpy as np
import pandas as pd
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------
# LOAD MODEL
# -----------------------------------

print("\nLoading embedding model...\n")

model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

# -----------------------------------
# LOAD EMBEDDINGS
# -----------------------------------

print("\nLoading candidate embeddings...\n")

with open(
    "candidate_embeddings.pkl",
    "rb"
) as file:

    data = pickle.load(file)

candidate_ids = data["candidate_ids"]
candidate_embeddings = data["embeddings"]
candidate_metadata = data["metadata"]

print(
    f"Loaded {len(candidate_ids)} candidates."
)

# -----------------------------------
# REAL REDROB JOB DESCRIPTION
# -----------------------------------

job_description = """
Senior AI Engineer — Founding Team at Redrob AI.

We need an engineer with deep technical depth in:
- embeddings
- retrieval systems
- ranking systems
- LLMs
- fine-tuning
- hybrid search
- vector databases
- semantic search infrastructure

The candidate should also demonstrate:
- strong Python engineering
- production ML deployment experience
- evaluation framework design for ranking systems
- hands-on experience with retrieval quality and ranking optimization
- experience shipping ranking/search/recommendation systems to real users
- product engineering mindset
- startup/product-company experience
- strong recruiter responsiveness

Preferred experience includes:
- vector databases
- recommendation systems
- search infrastructure
- marketplace/recruiting platforms

Candidates with only AI buzzwords
without production retrieval/ranking
experience should be down-weighted.
"""

# -----------------------------------
# JOB EMBEDDING
# -----------------------------------

print("\nGenerating job embedding...\n")

jd_embedding = model.encode(
    [job_description]
)

# -----------------------------------
# SKILL EXTRACTION
# -----------------------------------

def extract_skills(text):

    words = re.findall(
        r'\b[a-zA-Z0-9\-\+\.#]+\b',
        text.lower()
    )

    return set(words)

jd_skills = extract_skills(
    job_description
)

# -----------------------------------
# RANK CANDIDATES
# -----------------------------------

results = []

print("\nRanking candidates...\n")

for i in range(len(candidate_ids)):

    candidate_embedding = np.array(
        candidate_embeddings[i]
    ).reshape(1, -1)

    similarity = cosine_similarity(
        jd_embedding,
        candidate_embedding
    )[0][0]

    semantic_score = round(
        float(similarity * 100),
        2
    )

    candidate = candidate_metadata[i]

    signals = candidate.get(
        "redrob_signals",
        {}
    )

    # -----------------------------------
    # TRUST SCORE
    # -----------------------------------

    response_rate = signals.get(
        "recruiter_response_rate",
        0
    )

    github_score = signals.get(
        "github_activity_score",
        0
    )

    interview_rate = signals.get(
        "interview_completion_rate",
        0
    )

    profile_score = signals.get(
        "profile_completeness_score",
        0
    )

    trust_score = (

        (response_rate * 100) * 0.35 +
        github_score * 0.25 +
        (interview_rate * 100) * 0.25 +
        profile_score * 0.15

    )

    trust_score = round(
        trust_score,
        2
    )

    # -----------------------------------
    # EXPERIENCE SCORE
    # -----------------------------------

    experiences = candidate.get(
        "experience",
        []
    )

    relevant_experience_months = 0

    relevant_keywords = [

        "ai",
        "ml",
        "machine learning",
        "nlp",
        "deep learning",
        "python",
        "retrieval",
        "ranking",
        "search",
        "recommendation"

    ]

    for exp in experiences:

        title = exp.get(
            "title",
            ""
        ).lower()

        duration = exp.get(
            "duration_months",
            0
        )

        for keyword in relevant_keywords:

            if keyword in title:

                relevant_experience_months += duration
                break

    experience_score = min(
        (relevant_experience_months / 12) * 5,
        25
    )

    # -----------------------------------
    # SKILL MATCH SCORE
    # -----------------------------------

    skills = candidate.get(
        "skills",
        []
    )

    candidate_skill_names = []

    for skill in skills:

        skill_name = skill.get(
            "name",
            ""
        ).lower()

        candidate_skill_names.append(
            skill_name
        )

    candidate_skill_set = set(
        candidate_skill_names
    )

    matched_skills = jd_skills.intersection(
        candidate_skill_set
    )

    skill_match_score = min(
        len(matched_skills) * 5,
        20
    )

    # -----------------------------------
    # SUSPICION PENALTY
    # -----------------------------------

    suspicion_penalty = 0

    total_skills = len(
        candidate_skill_set
    )

    if total_skills > 40:

        suspicion_penalty += 10

    elif total_skills > 25:

        suspicion_penalty += 5

    # -----------------------------------
    # FINAL SCORE
    # -----------------------------------

    base_score = (

        semantic_score * 0.65 +
        trust_score * 0.20 +
        experience_score * 0.10 +
        skill_match_score * 0.05

    )

    final_score = (
        base_score -
        suspicion_penalty
    )

    final_score = max(
        final_score,
        0
    )

    final_score = round(
        final_score,
        2
    )

    # -----------------------------------
    # CLEAN REASONING
    # -----------------------------------

    if semantic_score >= 45:

        reasoning = (
            "Exceptional Retrieval AI Match"
        )

    elif semantic_score >= 40:

        reasoning = (
            "Strong AI Retrieval Match"
        )

    elif semantic_score >= 35:

        reasoning = (
            "Relevant AI Engineering Profile"
        )

    elif trust_score >= 80:

        reasoning = (
            "Excellent Recruiter Signals"
        )

    elif experience_score >= 15:

        reasoning = (
            "Strong Production AI Experience"
        )

    else:

        reasoning = (
            "Moderately Aligned Candidate"
        )

    # -----------------------------------
    # STORE RESULTS
    # -----------------------------------

    results.append({

        "candidate_id":
        candidate_ids[i],

        "score":
        final_score,

        "reasoning":
        reasoning

    })

# -----------------------------------
# CREATE DATAFRAME
# -----------------------------------

df = pd.DataFrame(results)

# -----------------------------------
# SORT CORRECTLY
# -----------------------------------

df = df.sort_values(

    by=[
        "score",
        "candidate_id"
    ],

    ascending=[
        False,
        True
    ]

)

# -----------------------------------
# RESET INDEX
# -----------------------------------

df.reset_index(
    drop=True,
    inplace=True
)

# -----------------------------------
# ADD RANK
# -----------------------------------

df.insert(
    1,
    "rank",
    range(1, len(df) + 1)
)

# -----------------------------------
# KEEP TOP 100
# -----------------------------------

top_100 = df.head(100)

# -----------------------------------
# REORDER COLUMNS
# -----------------------------------

top_100 = top_100[

    [
        "candidate_id",
        "rank",
        "score",
        "reasoning"
    ]

]

# -----------------------------------
# EXPORT CSV
# -----------------------------------

top_100.to_csv(
    "submission.csv",
    index=False
)

# -----------------------------------
# SUCCESS MESSAGE
# -----------------------------------

print(
    "\nTop 100 candidates saved!"
)

print(
    "\nFile created:"
)

print(
    "submission.csv"
)

# -----------------------------------
# SHOW TOP 10
# -----------------------------------

print("\nTop 10 Candidates:\n")

print(
    top_100.head(10)
)