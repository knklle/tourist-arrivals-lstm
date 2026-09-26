"""Shared presentation and workflow navigation for the forecasting lab."""
import streamlit as st

STEPS = [
    ("Dataset", "pages/1_Dataset.py", "raw_df", "Review the monthly arrivals data and check its coverage."),
    ("Clean", "pages/2_Clean.py", "clean_report", "Check duplicates, missing values, and unusual observations."),
    ("Features", "pages/3_Features.py", "feature_report", "Choose useful predictors using training data only."),
    ("Prepare", "pages/4_Prepare.py", "prepare_report", "Split data by date and create scaled monthly sequences."),
    ("Train", "pages/5_Train.py", "train_report", "Compare model settings and train your forecasting model."),
    ("Evaluate", "pages/6_Evaluate.py", "evaluate_report", "Check how well the model predicts held-out observations."),
    ("Explain", "pages/7_Explain.py", "explain_report", "Explore how each predictor contributes to model predictions."),
    ("Forecast", "pages/8_Forecast.py", "forecast_report", "Edit a monthly input window to explore a one-month forecast."),
]


def page_header(number):
    name, _, _, description = STEPS[number - 1]
    st.caption(f"PHILIPPINE TOURISM / STEP {number} OF 8")
    st.title(name if number != 3 else "Select predictors")
    st.write(description)
    left, right = st.columns(2)
    with left:
        if number > 1:
            previous = STEPS[number - 2]
            st.page_link(previous[1], label=f"← {previous[0]}")
        else:
            st.page_link("overview.py", label="← Overview")
    with right:
        if number < 8:
            following = STEPS[number]
            st.page_link(following[1], label=f"Next: {following[0]} →")
    if number > 1:
        # Evaluation and explanations are optional for generating a forecast.
        prerequisites = STEPS[:min(number - 1, 5)]
        missing = next((step for step in prerequisites if step[2] not in st.session_state), None)
        if missing:
            st.info(f"Before you begin, complete {missing[0]}. Your work stays available as you move between pages in this session.")
            st.page_link(missing[1], label=f"Go to {missing[0]}", icon="↗️")
    with st.expander("How this step fits into the workflow"):
        st.write(description)
        st.caption("Complete Dataset → Clean → Features → Prepare → Train in order. Then evaluate, explain, or forecast. Re-running an earlier step clears dependent results.")
    st.divider()


def page_footer(number):
    if STEPS[number - 1][2] not in st.session_state:
        return
    st.divider()
    if number < 8:
        following = STEPS[number]
        st.success(f"Results are available. Review them above, then continue to {following[0]}.")
        st.page_link(following[1], label=f"Continue to {following[0]} →", icon="➡️")
    else:
        st.page_link("pages/6_Evaluate.py", label="Review model accuracy", icon="📊")


def session_progress():
    completed = sum(key in st.session_state for _, _, key, _ in STEPS)
    with st.sidebar:
        st.divider()
        st.markdown("**Your session**")
        st.progress(completed / len(STEPS), text=f"{completed} of 8 steps have results")
        st.caption("Results last for this browser session. Download any results you want to keep.")
        with st.expander("Step status", expanded=False):
            for name, _, key, _ in STEPS:
                st.write(f"{'✓' if key in st.session_state else '○'} {name}")
