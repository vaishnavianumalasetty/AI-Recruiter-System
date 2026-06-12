import json
import streamlit as st
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(

    page_title="AI Recruiter System",
    page_icon="🤖",
    layout="wide"

)

# -----------------------------------
# TITLE
# -----------------------------------

st.title(
    "🤖 AI Recruiter Ranking System"
)

st.success(
    "Semantic AI-powered candidate ranking system built for Redrob AI Hiring Challenge"
)

st.markdown(
    """
    ### Intelligent Candidate Ranking using:
    
    ✅ Semantic AI Retrieval  
    ✅ Behavioral Trust Signals  
    ✅ Hybrid Ranking Intelligence  
    ✅ Recruiter-Aware Scoring
    """
)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

@st.cache_resource

def load_model():

    model = SentenceTransformer(
        'all-MiniLM-L6-v2'
    )

    return model

model = load_model()

# -----------------------------------
# LOAD DATASET
# -----------------------------------

@st.cache_data

def load_candidates():

    with open(
        "sample_candidates.json",
        "r",
        encoding="utf-8"
    ) as file:

        candidates = json.load(file)

    return candidates

candidates = load_candidates()

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.header(
    "📊 System Information"
)

st.sidebar.info(
    f"""
    Total Sample Candidates: {len(candidates)}
    
    Model: all-MiniLM-L6-v2
    
    Ranking Type:
    Hybrid Semantic + Behavioral
    """
)

# -----------------------------------
# JOB DESCRIPTION INPUT
# -----------------------------------

job_description = st.text_area(

    "📌 Enter Job Description",

    placeholder="""
Example:

Looking for an AI Engineer with experience in:
- Retrieval systems
- Vector databases
- LLMs
- Semantic search
- Python backend systems
- Recommendation engines
""",

    height=250
)

# -----------------------------------
# RANKING BUTTON
# -----------------------------------

if st.button("🚀 Rank Candidates"):

    if job_description.strip() == "":

        st.warning(
            "Please enter a Job Description."
        )

    else:

        with st.spinner(
            "Ranking candidates using Semantic AI..."
        ):

            # -----------------------------------
            # GENERATE JD EMBEDDING
            # -----------------------------------

            jd_embedding = model.encode(
                [job_description]
            )

            results = []

            # -----------------------------------
            # PROCESS CANDIDATES
            # -----------------------------------

            for candidate in candidates:

                candidate_text = ""

                # -----------------------------------
                # SKILLS
                # -----------------------------------

                for skill in candidate.get(
                    "skills",
                    []
                ):

                    candidate_text += (
                        " " +
                        skill.get("name", "")
                    )

                # -----------------------------------
                # EXPERIENCE
                # -----------------------------------

                for exp in candidate.get(
                    "experience",
                    []
                ):

                    candidate_text += (
                        " " +
                        exp.get("title", "")
                    )

                # -----------------------------------
                # CANDIDATE EMBEDDING
                # -----------------------------------

                candidate_embedding = model.encode(
                    [candidate_text]
                )

                # -----------------------------------
                # SEMANTIC SIMILARITY
                # -----------------------------------

                similarity = cosine_similarity(

                    jd_embedding,
                    candidate_embedding

                )[0][0]

                semantic_score = round(

                    float(similarity * 100),
                    2

                )

                # -----------------------------------
                # BEHAVIORAL SIGNALS
                # -----------------------------------

                signals = candidate.get(
                    "redrob_signals",
                    {}
                )

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

                behavioral_score = (

                    (response_rate * 100) * 0.30 +
                    github_score * 0.20 +
                    (interview_rate * 100) * 0.30 +
                    profile_score * 0.20

                )

                behavioral_score = round(
                    behavioral_score,
                    2
                )

                # -----------------------------------
                # FINAL SCORE
                # -----------------------------------

                final_score = (

                    semantic_score * 0.70 +
                    behavioral_score * 0.30

                )

                final_score = round(
                    final_score,
                    2
                )

                # -----------------------------------
                # REASONING
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

                elif behavioral_score >= 75:

                    reasoning = (
                        "Excellent Recruiter Signals"
                    )

                else:

                    reasoning = (
                        "Moderately Aligned Candidate"
                    )

                # -----------------------------------
                # STORE RESULTS
                # -----------------------------------

                results.append({

                    "Candidate ID":
                    candidate.get(
                        "candidate_id"
                    ),

                    "Semantic Score":
                    semantic_score,

                    "Behavioral Score":
                    behavioral_score,

                    "Final Score":
                    final_score,

                    "Reasoning":
                    reasoning
                })

            # -----------------------------------
            # CREATE DATAFRAME
            # -----------------------------------

            df = pd.DataFrame(results)

            # -----------------------------------
            # SORT RESULTS
            # -----------------------------------

            df = df.sort_values(

                by=[
                    "Final Score",
                    "Candidate ID"
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
            # ADD RANK COLUMN
            # -----------------------------------

            df.insert(

                0,
                "Rank",
                range(1, len(df) + 1)

            )

            # -----------------------------------
            # TOP CANDIDATES
            # -----------------------------------

            top_candidates = df.head(100)

            # -----------------------------------
            # METRICS
            # -----------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(

                    label="Candidates Ranked",
                    value=len(df)

                )

            with col2:

                st.metric(

                    label="Top Score",
                    value=top_candidates[
                        "Final Score"
                    ].max()

                )

            with col3:

                st.metric(

                    label="Displayed Results",
                    value=20

                )

            # -----------------------------------
            # DISPLAY RESULTS
            # -----------------------------------

            st.subheader(
                "🏆 Top Ranked Candidates"
            )

            st.dataframe(

                top_candidates,

                use_container_width=True

            )

            # -----------------------------------
            # DOWNLOAD BUTTON
            # -----------------------------------

            csv = top_candidates.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(

                label="📥 Download Ranked Candidates CSV",

                data=csv,

                file_name="ranked_candidates.csv",

                mime="text/csv"

            )

# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown("---")

st.caption(
    "Built using Semantic AI Retrieval + Hybrid Ranking Intelligence"
)