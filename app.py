"""Streamlit interface for EconoLens AI.

Author: Rishabh Shah
Advisor: Dr. Qingyang Xiao
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import streamlit as st

from usage_tracker import record_visit

from econolens_core import (
    ADVISOR_NAME,
    APP_NAME,
    APP_TAGLINE,
    ArticleMetadata,
    EconomicsNewsPipeline,
    FeedbackBandit,
    ReaderProfile,
    STUDENT_NAME,
    fetch_public_article,
)

st.set_page_config(
    page_title=APP_NAME,
    page_icon="E",
    layout="wide",
    initial_sidebar_state="expanded",
)

SYNTHETIC_ARTICLE = """
A fictional national statistics office reported that annual consumer inflation slowed from 4.2 percent to 3.6 percent, but rent, insurance, medical services, and restaurant prices remained high. The central bank kept its benchmark interest rate at 4.75 percent because officials wanted more evidence that core inflation was moving down in a lasting way. Policymakers said a future rate cut may be possible if price growth continues to slow and the job market remains stable.

Employers added 145,000 jobs during the month, although wage growth cooled and average weekly hours declined. The unemployment rate edged up to 4.4 percent as more people entered the labor force and began looking for work. Hiring remained strongest in health services, transportation, and local government, while several technology and manufacturing companies announced smaller recruiting plans.

Retail sales increased 0.5 percent, supported by online purchases and spending on services. However, lower-income households used more credit to cover essential expenses. Credit card delinquency increased among younger borrowers, and banks reported tighter loan standards. Surveys showed that many consumers remained worried about food, housing, and insurance bills even as the overall inflation rate improved.

In housing, mortgage rates near 6.8 percent limited affordability even though the number of homes for sale improved. Apartment rent growth slowed in cities where new construction added supply, while other areas still faced shortages. Builders said lower material costs helped some projects, but labor and financing costs remained high.

Manufacturers reported shorter delivery times and lower shipping costs after supply chain conditions improved. A weaker currency, however, raised the price of imported fuel, electronics, and industrial equipment. Exporters benefited from more competitive prices abroad, while retailers that depend on imported products faced pressure on profit margins.

Economists described the outlook as a possible soft landing, but they warned that energy prices, trade disruptions, or renewed wage pressure could delay interest-rate cuts. For households, the story is mixed: inflation is slowing, yet many bills remain high, borrowing is still expensive, and job conditions are becoming less uniformly strong. The article therefore suggests watching the next inflation report, hiring data, consumer debt, and central-bank guidance rather than relying on a single headline.
""".strip()


@st.cache_data(show_spinner=False)
def article_from_url(url: str) -> dict[str, str]:
    return fetch_public_article(url)


def get_pipeline() -> EconomicsNewsPipeline:
    if "econolens_pipeline" not in st.session_state:
        st.session_state.econolens_pipeline = EconomicsNewsPipeline(feedback_path=None)
    return st.session_state.econolens_pipeline


pipeline = get_pipeline()

# Record one use per Streamlit browser session. Streamlit reruns the script
# for widget interactions, so session_state prevents double-counting.
if "usage_recorded" not in st.session_state:
    st.session_state.usage_recorded = True
    st.session_state.usage_total, st.session_state.usage_backend = record_visit(st.secrets)

st.markdown(
    """
    <style>
    .block-container {padding-top: 5.2rem; padding-bottom: 3rem; max-width: 1400px;}
    .econolens-credit {
        border: 1px solid rgba(49, 51, 63, 0.18);
        border-radius: 0.75rem;
        padding: 0.8rem 1rem;
        margin: 0.35rem 0 1rem 0;
        background: rgba(240, 242, 246, 0.45);
    }
    .econolens-credit strong {font-weight: 700;}
    .econolens-kicker {
        letter-spacing: 0.12em;
        font-size: 0.78rem;
        font-weight: 700;
        opacity: 0.72;
        margin-top: 0.35rem; margin-bottom: 0.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="econolens-kicker">AI ECONOMICS EDUCATION</div>', unsafe_allow_html=True)
st.title(APP_NAME)
st.caption(APP_TAGLINE)
usage_col1, usage_col2 = st.columns([5, 1])
with usage_col1:
    st.markdown(
        f'<div class="econolens-credit"><strong>Author:</strong> {STUDENT_NAME}'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;<strong>Advisor:</strong> {ADVISOR_NAME}</div>',
        unsafe_allow_html=True,
    )
with usage_col2:
    st.metric("App uses", f"{st.session_state.usage_total:,}")
st.caption(f"Usage counter records one use per new Streamlit session. Backend: {st.session_state.usage_backend}.")
st.markdown(
    "Turn a difficult economics article into a layered explanation, a grounded story, "
    "and possible everyday-life impacts. The app is an educational prototype, not financial advice."
)

with st.sidebar:
    st.header("Reader profile")
    country = st.text_input("Country or region", "United States")
    age_group = st.selectbox("Age group", ["Under 18", "18-24", "25-34", "35-49", "50-64", "65+"])
    role = st.selectbox(
        "Current role",
        ["Student", "Employed", "Job seeker", "Business owner", "Retired", "Other"],
    )
    income_situation = st.selectbox(
        "Income situation",
        ["Lower income", "Middle income", "Higher income", "Prefer not to say"],
        index=1,
    )
    housing_status = st.selectbox(
        "Housing status",
        ["Renter", "Homeowner", "Planning to buy", "Living with family", "Other"],
    )
    debt_exposure = st.selectbox(
        "Debt exposure",
        [
            "No debt",
            "Some student or consumer debt",
            "Mortgage debt",
            "High variable-rate debt",
            "Prefer not to say",
        ],
        index=1,
    )
    budget_sensitivity = st.selectbox("Budget sensitivity", ["Low", "Moderate", "High"], index=1)
    interests_text = st.text_input(
        "Topics you care about",
        "everyday prices, jobs, rent, student loans",
    )
    use_neural = st.checkbox(
        "Enable optional FLAN-T5 rewrite",
        value=False,
        help=(
            "This downloads an open-source neural model and can be slower. "
            "The evidence-based deterministic analysis works without it. "
            "For a default Streamlit Cloud deployment, install requirements-neural.txt only if this feature is needed."
        ),
    )
    st.divider()
    st.subheader("Project credits")
    st.markdown(f"**Author:** {STUDENT_NAME}")
    st.markdown(f"**Advisor:** {ADVISOR_NAME}")
    st.caption("Senior student AI and economics project")

profile = ReaderProfile.from_interests_text(
    interests_text,
    country=country,
    age_group=age_group,
    role=role,
    income_situation=income_situation,
    housing_status=housing_status,
    debt_exposure=debt_exposure,
    budget_sensitivity=budget_sensitivity,
)

st.subheader("1. Add an economics article")
input_mode = st.radio(
    "Input method",
    ["Synthetic demonstration", "Paste article text", "Public article URL", "Upload text file"],
    horizontal=True,
)

article_text = ""
metadata = ArticleMetadata()

if input_mode == "Synthetic demonstration":
    article_text = SYNTHETIC_ARTICLE
    metadata = ArticleMetadata(
        title="Synthetic economics article for demonstration",
        source="EconoLens fictional example",
    )
    st.info("The demonstration article is fictional and is included only to test the workflow.")
    with st.expander("View demonstration article"):
        st.write(article_text)
elif input_mode == "Paste article text":
    title = st.text_input("Article title", "Untitled economics article")
    source = st.text_input("Publisher or source", "User-provided text")
    article_text = st.text_area(
        "Paste at least about 80 words",
        height=300,
        placeholder="Paste the article here...",
    )
    metadata = ArticleMetadata(title=title, source=source)
elif input_mode == "Public article URL":
    url = st.text_input("Public URL")
    metadata = ArticleMetadata(title="Web article", source="URL", url=url)
    if url:
        st.caption("The extractor reads public webpage text only and does not bypass paywalls or access controls.")
else:
    uploaded = st.file_uploader("Upload TXT, MD, or HTML", type=["txt", "md", "html", "htm"])
    if uploaded is not None:
        raw = uploaded.getvalue().decode("utf-8", errors="replace")
        if uploaded.name.lower().endswith((".html", ".htm")):
            from bs4 import BeautifulSoup

            raw = BeautifulSoup(raw, "html.parser").get_text(" ")
        article_text = raw
        metadata = ArticleMetadata(title=uploaded.name, source="Uploaded file")

analyze_clicked = st.button("Analyze article", type="primary", use_container_width=True)
if analyze_clicked:
    try:
        if input_mode == "Public article URL":
            if not metadata.url:
                raise ValueError("Enter a public article URL.")
            with st.spinner("Extracting public article text..."):
                fetched = article_from_url(metadata.url)
            article_text = fetched["text"]
            metadata = ArticleMetadata(
                title=fetched["title"],
                source=fetched["source"],
                url=fetched["url"],
            )
        with st.spinner("Building the economic hierarchy and personal-impact analysis..."):
            result = pipeline.analyze(
                article_text,
                profile=profile,
                metadata=metadata,
                use_neural=use_neural,
            )
        st.session_state.econolens_result = result
        st.success("Analysis complete.")
    except Exception as exc:
        st.error(str(exc))

result = st.session_state.get("econolens_result")
if result:
    st.divider()
    st.subheader("2. Explore the explanation")
    tabs = st.tabs(
        [
            "Simple hierarchy",
            "Economic story",
            "Personal impact",
            "Evidence and terms",
            "Word forecast",
            "Quality",
            "Feedback learning",
        ]
    )

    with tabs[0]:
        hierarchy = result["hierarchy"]
        st.info(hierarchy["Level 1 - 15-second takeaway"])
        st.markdown("#### Main developments")
        for item in hierarchy["Level 2 - Main developments"]:
            st.markdown(f"- {item}")

        st.markdown("#### Main economic themes")
        theme_df = pd.DataFrame(hierarchy["Level 3 - Main economic themes"])
        st.dataframe(theme_df, use_container_width=True, hide_index=True)

        topic_df = pd.DataFrame(
            [{"topic": topic, "share": score} for topic, score in result["topic_scores"].items()]
        ).set_index("topic")
        st.bar_chart(topic_df)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Causes stated in the article")
            for item in hierarchy["Level 4 - Causes and evidence"]["possible_drivers_stated_in_article"]:
                st.markdown(f"- {item}")
        with col2:
            st.markdown("#### What the article says may happen next")
            for item in hierarchy["Level 5 - People and next chapter"]["what_the_article_says_may_happen_next"]:
                st.markdown(f"- {item}")

        if result.get("neural"):
            st.markdown("#### Optional neural rewrite")
            neural = result["neural"]
            if neural.get("error"):
                st.warning(neural["error"])
            else:
                st.write(neural["plain_summary"])
                if neural.get("unsupported_numbers_detected"):
                    st.warning(
                        "Number-like text not found in the extractive notes: "
                        + ", ".join(neural["unsupported_numbers_detected"])
                    )
                st.caption(neural["grounding_note"])

    with tabs[1]:
        for label, paragraph in result["story"].items():
            st.markdown(f"#### {label}")
            st.write(paragraph)
        neural = result.get("neural") or {}
        if neural.get("story"):
            with st.expander("Compare with optional neural story"):
                st.write(neural["story"])
                st.caption(neural["grounding_note"])

    with tabs[2]:
        st.caption(
            "These are plausible pathways based on the article and profile. They are not guaranteed outcomes or individualized financial advice."
        )
        for impact in result["personal_impacts"]:
            with st.expander(f"{impact['area']} - confidence {impact['confidence']:.0%}", expanded=True):
                st.markdown(f"**Possible impact:** {impact['possible_impact']}")
                st.markdown(f"**Why:** {impact['why']}")
                st.markdown(f"**Practical check:** {impact['practical_check']}")
                st.markdown(f"**Article evidence:** {impact['article_evidence']}")
                st.progress(float(impact["confidence"]))

    with tabs[3]:
        st.markdown("#### Extractive evidence summary")
        for sentence in result["source_summary"]:
            st.markdown(f"- {sentence}")
        st.markdown("#### Economics glossary")
        if result["terms"]:
            terms_df = pd.DataFrame(result["terms"])
            st.dataframe(
                terms_df[["term", "count", "plain_definition", "why_it_matters", "evidence"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No glossary terms were detected.")
        st.markdown("#### Key phrases")
        st.dataframe(pd.DataFrame(result["key_phrases"]), use_container_width=True, hide_index=True)

    with tabs[4]:
        st.caption(
            "This transparent n-gram model predicts words from patterns in the current article. "
            "It is a forecasting demonstration, not a language model claim about future economic events."
        )
        default_prompts = list(result["forecast_examples"])
        prompt_choice = st.selectbox(
            "Try a phrase from the article",
            options=default_prompts or ["interest rates"],
        )
        custom_prefix = st.text_input("Or enter your own prefix", prompt_choice)
        if st.button("Forecast next words"):
            forecast_rows = pipeline.word_forecaster.predict(custom_prefix, top_k=8)
            st.dataframe(pd.DataFrame(forecast_rows), use_container_width=True, hide_index=True)
            st.markdown(
                "**Deterministic autocomplete example:** "
                + pipeline.word_forecaster.autocomplete(custom_prefix, additional_words=8)
            )

    with tabs[5]:
        st.markdown("#### Readability comparison")
        readability_df = pd.DataFrame(result["readability"])
        st.dataframe(readability_df, use_container_width=True, hide_index=True)
        st.markdown("#### Quality and safety notes")
        for flag in result["quality_flags"]:
            st.markdown(f"- {flag}")
        with st.expander("Technical model details"):
            st.write(f"Topic DNN: {pipeline.topic_model.architecture}")
            st.write(f"Training iterations: {pipeline.topic_model.training_iterations}")
            st.write(
                "Feedback component: an epsilon-greedy multi-armed bandit that learns presentation preference, not economic facts."
            )

    with tabs[6]:
        st.caption(
            "Feedback changes which format is recommended. It does not modify evidence, numbers, or extracted claims."
        )
        with st.form("feedback_form"):
            style = st.selectbox("Explanation style", FeedbackBandit.STYLES)
            overall = st.slider("Overall rating", 1, 5, 4)
            clarity = st.slider("Clarity", 1, 5, 4)
            usefulness = st.slider("Usefulness", 1, 5, 4)
            comment = st.text_area("Optional comment")
            submitted = st.form_submit_button("Submit feedback")
        if submitted:
            record = pipeline.feedback.update(style, overall, clarity, usefulness, comment)
            st.success(f"Feedback recorded. Reward = {record['reward']:.3f}")
        st.dataframe(pipeline.feedback.status(), use_container_width=True, hide_index=True)
        st.write(f"Current recommended style: **{pipeline.feedback.recommend(epsilon=0.0)}**")

    st.divider()
    download_col1, download_col2 = st.columns(2)
    with download_col1:
        st.download_button(
            "Download Markdown report",
            data=result["markdown_report"],
            file_name="econolens_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with download_col2:
        st.download_button(
            "Download analysis JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name="econolens_analysis.json",
            mime="application/json",
            use_container_width=True,
        )

st.divider()
st.caption(
    f"Author: {STUDENT_NAME} | Advisor: {ADVISOR_NAME} | Educational prototype | "
    "Use public content lawfully and verify important claims against the original source and current official data."
)
