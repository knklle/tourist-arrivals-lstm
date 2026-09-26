import numpy as np
import pandas as pd
import streamlit as st
from ui import page_header, page_footer


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

page_header(8)

st.write(
    "Use the training window below as a starting point. "
    "The prediction is for the month immediately after your input window."
)


# --------------------------------------------------
# 1. CHECK PREVIOUS STEPS
# --------------------------------------------------

required_keys = [
    "model",
    "X_train_seq",
    "scaler_X",
    "scaler_y",
    "selected_features",
]

missing_keys = [
    key for key in required_keys
    if key not in st.session_state
]

if missing_keys:

    st.warning(
        "The trained model or prepared data is missing. "
        "Please complete Prepare and Train first."
    )

    st.stop()


# --------------------------------------------------
# 2. LOAD MODEL COMPONENTS
# --------------------------------------------------

model = st.session_state.model

scaler_X = st.session_state.scaler_X

scaler_y = st.session_state.scaler_y

features = list(
    st.session_state.selected_features
)

X_train_seq = np.asarray(
    st.session_state.X_train_seq,
    dtype=np.float32
)

lookback = X_train_seq.shape[1]

n_features = X_train_seq.shape[2]


# --------------------------------------------------
# 3. VALIDATE FEATURE COUNT
# --------------------------------------------------

if len(features) != n_features:

    st.error(
        "The number of selected features does not "
        "match the trained LSTM input shape."
    )

    st.stop()


# --------------------------------------------------
# 4. PREPARE DEFAULT INPUT WINDOW
# --------------------------------------------------

last_window_scaled = (
    X_train_seq[-1]
)

last_window_original = (
    scaler_X.inverse_transform(
        last_window_scaled
    )
)

default_df = pd.DataFrame(
    last_window_original,
    columns=features
)


# --------------------------------------------------
# 5. DISPLAY MODEL INFORMATION
# --------------------------------------------------

st.subheader("1. Forecast Input")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Lookback",
        f"{lookback} months"
    )

with col2:

    st.metric(
        "Predictors",
        n_features
    )

st.write(
    "The table below is pre-filled using the "
    "latest valid training window, which may precede the most recent dataset observations. "
    "Rows run from oldest to newest; keep this order when editing."
)


# --------------------------------------------------
# 6. EDITABLE INPUT TABLE
# --------------------------------------------------

edited_df = st.data_editor(
    default_df.rename_axis("Month in window").set_axis(range(1, lookback + 1)),
    num_rows="fixed",
    use_container_width=True,
    key="forecast_editor"
)


# --------------------------------------------------
# 7. VALIDATE INPUT
# --------------------------------------------------

if len(edited_df) != lookback:

    st.error(
        f"The forecast requires exactly "
        f"{lookback} monthly rows."
    )

    st.stop()


missing_input_columns = [
    feature
    for feature in features
    if feature not in edited_df.columns
]

if missing_input_columns:

    st.error(
        f"Missing forecast columns: "
        f"{missing_input_columns}"
    )

    st.stop()


# --------------------------------------------------
# 8. FORECAST
# --------------------------------------------------

if st.button(
    "Forecast Next Month",
    type="primary"
):

    forecast_df = (
        edited_df[
            features
        ]
        .copy()
    )


    # Convert values to numeric.

    for feature in features:

        forecast_df[
            feature
        ] = pd.to_numeric(
            forecast_df[feature],
            errors="coerce"
        )


    # Check for missing/non-numeric entries.

    if not np.isfinite(forecast_df.to_numpy(dtype=float)).all():

        st.error(
            "The forecast input contains missing "
            "non-numeric, or infinite values."
        )

        st.stop()


    # --------------------------------------------------
    # SCALE INPUT USING TRAINING-FITTED SCALER
    # --------------------------------------------------

    scaled_input = (
        scaler_X.transform(
            forecast_df
        )
    )


    model_input = (
        scaled_input
        .reshape(
            1,
            lookback,
            n_features
        )
        .astype(
            np.float32
        )
    )


    # --------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------

    prediction_scaled = (
        model.predict(
            model_input,
            verbose=0
        )
    )


    prediction = (
        scaler_y
        .inverse_transform(
            prediction_scaled
        )
        .reshape(-1)[0]
    )


    st.session_state.forecast_input = edited_df.copy()

    st.session_state.forecast_report = (
        float(prediction)
    )


# --------------------------------------------------
# 9. DISPLAY FORECAST
# --------------------------------------------------

if "forecast_report" in st.session_state and edited_df.equals(st.session_state.get("forecast_input")):

    prediction = (
        st.session_state.forecast_report
    )

    st.subheader(
        "2. Forecast Result"
    )


    st.metric(
        "Predicted Tourist Arrivals",
        f"{prediction:,.0f}"
    )


    st.success(
        "Forecast generated successfully."
    )


else:

    st.info(
        "Edit the predictor table if needed, "
        "then click Forecast Next Month."
    )

page_footer(8)
