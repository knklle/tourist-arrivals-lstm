import numpy as np
import pandas as pd
import streamlit as st
from ui import page_header, page_footer


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

page_header(6)

st.write(
    "Evaluate the trained LSTM on the held-out test set "
    "using MAE, RMSE, MAPE, and R², then compare it "
    "with naive and seasonal-naive baselines."
)


# --------------------------------------------------
# 1. CHECK PREVIOUS STEPS
# --------------------------------------------------

required_keys = [
    "model",
    "X_test_seq",
    "y_test_seq",
    "scaler_y",
    "test_df",
    "test_target_indices",
    "test_target_dates",
]

missing_keys = [
    key for key in required_keys
    if key not in st.session_state
]

if missing_keys:

    st.warning(
        "Evaluation data or the trained model is missing. "
        "Please complete the Prepare and Train pages first."
    )

    st.stop()


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

model = st.session_state.model

X_test = np.asarray(
    st.session_state.X_test_seq,
    dtype=np.float32
)

y_test_scaled = np.asarray(
    st.session_state.y_test_seq,
    dtype=np.float32
)

scaler_y = st.session_state.scaler_y

test_df = (
    st.session_state.test_df
    .copy()
    .reset_index(drop=True)
)

target_indices = list(
    st.session_state.test_target_indices
)

target_dates = pd.to_datetime(
    st.session_state.test_target_dates
)


# --------------------------------------------------
# 3. VALIDATE ALIGNMENT
# --------------------------------------------------

if len(X_test) != len(target_indices):

    st.error(
        "Test sequence count does not match the saved "
        "target indices. Please rerun the Prepare page."
    )

    st.stop()


if len(X_test) != len(target_dates):

    st.error(
        "Test sequence count does not match the saved "
        "target dates. Please rerun the Prepare page."
    )

    st.stop()


# --------------------------------------------------
# 4. DISPLAY TEST INFORMATION
# --------------------------------------------------

st.subheader("1. Held-Out Test Set")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Test Windows",
        len(X_test)
    )

with col2:

    st.metric(
        "Lookback",
        f"{X_test.shape[1]} months"
    )

with col3:

    st.metric(
        "Predictors",
        X_test.shape[2]
    )


if len(target_dates) > 0:

    st.write(
        "Forecast target period:",
        target_dates.min().strftime("%Y-%m"),
        "to",
        target_dates.max().strftime("%Y-%m")
    )


st.info(
    "These observations were not used for feature "
    "selection, hyperparameter tuning, or model fitting."
)


# --------------------------------------------------
# 5. METRIC FUNCTION
# --------------------------------------------------

def score_model(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float
    ).reshape(-1)

    predicted = np.asarray(
        predicted,
        dtype=float
    ).reshape(-1)


    mae = float(
        np.mean(
            np.abs(
                actual - predicted
            )
        )
    )


    rmse = float(
        np.sqrt(
            np.mean(
                (
                    actual - predicted
                ) ** 2
            )
        )
    )


    # Avoid division by zero in MAPE.
    nonzero_mask = (
        actual != 0
    )

    if np.any(nonzero_mask):

        mape = float(
            np.mean(
                np.abs(
                    (
                        actual[nonzero_mask]
                        - predicted[nonzero_mask]
                    )
                    / actual[nonzero_mask]
                )
            ) * 100
        )

    else:

        mape = np.nan


    ss_res = float(
        np.sum(
            (
                actual - predicted
            ) ** 2
        )
    )


    ss_tot = float(
        np.sum(
            (
                actual
                - np.mean(actual)
            ) ** 2
        )
    )


    if ss_tot == 0:

        r2 = np.nan

    else:

        r2 = float(
            1 - (
                ss_res / ss_tot
            )
        )


    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "R²": r2,
    }


# --------------------------------------------------
# 6. RUN EVALUATION
# --------------------------------------------------

SEASONAL_PERIOD = 12


if st.button(
    "Evaluate on Test Set",
    type="primary"
):

    with st.spinner(
        "Evaluating the LSTM..."
    ):

        # ------------------------------------------
        # A. LSTM PREDICTIONS
        # ------------------------------------------

        pred_scaled = model.predict(
            X_test,
            verbose=0
        )


        predicted = (
            scaler_y
            .inverse_transform(
                pred_scaled
            )
            .reshape(-1)
        )


        actual = (
            scaler_y
            .inverse_transform(
                y_test_scaled
            )
            .reshape(-1)
        )


        # ------------------------------------------
        # B. ORIGINAL TEST ARRIVALS
        # ------------------------------------------

        test_arrivals = pd.to_numeric(
            test_df["arrivals"],
            errors="coerce"
        ).to_numpy(
            dtype=float
        )


        # ------------------------------------------
        # C. BUILD BASELINES
        # ------------------------------------------

        naive_predictions = []

        seasonal_predictions = []


        for idx in target_indices:

            # Naive:
            # predict the previous month's arrivals.
            naive_index = (
                idx - 1
            )

            # Seasonal naive:
            # predict arrivals 12 months earlier.
            seasonal_index = (
                idx - SEASONAL_PERIOD
            )


            if (
                naive_index < 0
                or seasonal_index < 0
            ):

                st.error(
                    "A baseline index falls before the "
                    "start of the test period."
                )

                st.stop()


            naive_predictions.append(
                test_arrivals[
                    naive_index
                ]
            )


            seasonal_predictions.append(
                test_arrivals[
                    seasonal_index
                ]
            )


        naive_predictions = np.asarray(
            naive_predictions,
            dtype=float
        )


        seasonal_predictions = np.asarray(
            seasonal_predictions,
            dtype=float
        )


        # ------------------------------------------
        # D. VALIDATE BASELINES
        # ------------------------------------------

        if (
            np.isnan(
                naive_predictions
            ).any()
            or np.isnan(
                seasonal_predictions
            ).any()
        ):

            st.error(
                "Missing arrival values were found "
                "inside a baseline calculation."
            )

            st.stop()


        # ------------------------------------------
        # E. CALCULATE METRICS
        # ------------------------------------------

        lstm_metrics = score_model(
            actual,
            predicted
        )


        naive_metrics = score_model(
            actual,
            naive_predictions
        )


        seasonal_metrics = score_model(
            actual,
            seasonal_predictions
        )


        metrics_df = pd.DataFrame({
            "LSTM": lstm_metrics,
            "Naive": naive_metrics,
            "Seasonal Naive": seasonal_metrics,
        })


        # ------------------------------------------
        # F. FORECAST COMPARISON TABLE
        # ------------------------------------------

        comparison_df = pd.DataFrame({
            "Date": target_dates,
            "Actual": actual,
            "LSTM Prediction": predicted,
            "Naive Prediction": naive_predictions,
            "Seasonal Naive Prediction": seasonal_predictions,
        })


        comparison_df = (
            comparison_df
            .sort_values("Date")
            .reset_index(drop=True)
        )


        # ------------------------------------------
        # G. SAVE RESULTS
        # ------------------------------------------

        st.session_state.evaluate_report = (
            metrics_df
        )


        st.session_state.evaluate_predictions = (
            comparison_df
        )


        st.success(
            "Test-set evaluation completed!"
        )


# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------

if "evaluate_report" in st.session_state:

    metrics_df = (
        st.session_state.evaluate_report
    )

    comparison_df = (
        st.session_state.evaluate_predictions
    )


    st.subheader(
        "2. Evaluation Metrics"
    )


    st.dataframe(
        metrics_df.style.format(
            "{:,.4f}"
        ),
        use_container_width=True
    )


    st.caption(
        "Lower MAE, RMSE, and MAPE indicate smaller "
        "forecast errors. R² describes how much variation "
        "in the held-out arrivals is represented by the predictions."
    )


    # --------------------------------------------------
    # 8. ACTUAL VS FORECAST
    # --------------------------------------------------

    st.subheader(
        "3. Actual vs. Forecast"
    )


    chart_df = (
        comparison_df[
            [
                "Date",
                "Actual",
                "LSTM Prediction",
                "Naive Prediction",
                "Seasonal Naive Prediction",
            ]
        ]
        .set_index("Date")
    )


    st.line_chart(
        chart_df
    )


    # --------------------------------------------------
    # 9. FORECAST TABLE
    # --------------------------------------------------

    st.subheader(
        "4. Test Predictions"
    )


    display_df = (
        comparison_df.copy()
    )


    numeric_columns = [
        "Actual",
        "LSTM Prediction",
        "Naive Prediction",
        "Seasonal Naive Prediction",
    ]


    display_df[
        numeric_columns
    ] = display_df[
        numeric_columns
    ].round(0)


    st.dataframe(
        display_df,
        use_container_width=True
    )


    st.success(
        "Evaluation finished. Keep these metrics "
        "for your laboratory interpretation."
    )


else:

    st.info(
        "Click Evaluate on Test Set to calculate "
        "the forecasting metrics."
    )

page_footer(6)
