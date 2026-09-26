import streamlit as st
from ui import STEPS

st.caption("PHILIPPINE TOURISM • FORECASTING LAB")
st.title("From monthly arrivals to informed forecasts")
st.write("Explore tourism patterns, build a forecasting model, and understand what drives its predictions. Follow the guided workflow below to get started.")
completed = sum(key in st.session_state for _, _, key, _ in STEPS)
next_step = next((step for step in STEPS if step[2] not in st.session_state), STEPS[-1])
with st.container(border=True):
    st.subheader("Continue your work" if completed else "Start with your dataset")
    st.write(next_step[3])
    st.page_link(next_step[1], label=f"Open {next_step[0]} →", icon="▶️")
    st.progress(completed / 8, text=f"{completed} of 8 steps have results")
a, b, c = st.columns(3)
a.metric("Data frequency", "Monthly")
b.metric("Forecast horizon", "1 month")
c.metric("Workflow", "8 steps")
st.subheader("Your forecasting workflow")
for title, steps in [("1 · Get the data ready", STEPS[:4]), ("2 · Build and understand", STEPS[4:])]:
    st.markdown(f"**{title}**")
    columns = st.columns(2)
    for i, (name, path, key, description) in enumerate(steps):
        with columns[i % 2], st.container(border=True):
            st.page_link(path, label=name, icon="✅" if key in st.session_state else "📋")
            st.write(description)
            st.caption("Results available" if key in st.session_state else "No results yet")
with st.expander("New to forecasting? Read the quick guide"):
    st.markdown("- **Predictors** are inputs such as weather and seasonal indicators.\n- **LSTM** is a model that learns patterns across a sequence of months.\n- **Evaluation** compares predictions with actual arrivals the model did not train on.\n- **SHAP** estimates how inputs influence the model’s predictions.")
    st.info("Finish the first five steps in order. Evaluate your model before relying on its forecasts. Results are held in your current session.")
