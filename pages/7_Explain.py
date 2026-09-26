import numpy as np
import pandas as pd
import streamlit as st
from ui import page_header, page_footer


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

page_header(7)

st.write(
    "Use SHAP to examine which predictors influenced "
    "the trained LSTM forecasts."
)


# --------------------------------------------------
# 1. CHECK PREVIOUS STEPS
# --------------------------------------------------

required_keys = [
    "model",
    "X_train_seq",
    "X_test_seq",
    "selected_features",
    "scaler_y",
]

missing_keys = [
    key for key in required_keys
    if key not in st.session_state
]

if missing_keys:

    st.warning(
        "The trained model or prepared sequences are "
        "missing. Please complete Prepare and Train first."
    )

    st.stop()


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

model = st.session_state.model

X_train = np.asarray(
    st.session_state.X_train_seq,
    dtype=np.float32
)

X_test = np.asarray(
    st.session_state.X_test_seq,
    dtype=np.float32
)

features = list(
    st.session_state.selected_features
)

scaler_y = (
    st.session_state.scaler_y
)

lookback = X_train.shape[1]
n_features = X_train.shape[2]


# --------------------------------------------------
# 3. DISPLAY MODEL INFORMATION
# --------------------------------------------------

st.subheader("1. Model Input")

col1, col2, col3 = st.columns(3)

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

with col3:

    st.metric(
        "Test Windows",
        len(X_test)
    )


st.write(
    "Selected predictors:"
)

st.write(
    features
)


# --------------------------------------------------
# 4. SHAP SETTINGS
# --------------------------------------------------

st.subheader("2. SHAP Settings")

st.write(
    "Kernel SHAP is used because it is model-agnostic "
    "and works with the LSTM through model.predict()."
)

st.info(
    "SHAP can take some time because many predictions "
    "must be generated to estimate feature contributions."
)


# --------------------------------------------------
# 5. RUN SHAP
# --------------------------------------------------

if st.button(
    "Compute SHAP Explanations",
    type="primary"
):

    import shap

    with st.spinner(
        "Computing SHAP values... this may take a few minutes."
    ):

        # ------------------------------------------
        # A. BLACK-BOX PREDICTION FUNCTION
        # ------------------------------------------

        def predict_flat(flat_data):

            flat_data = np.asarray(
                flat_data,
                dtype=np.float32
            )

            sequences = flat_data.reshape(
                -1,
                lookback,
                n_features
            )

            predictions_scaled = model.predict(
                sequences,
                verbose=0
            )

            predictions = (
                scaler_y
                .inverse_transform(
                    predictions_scaled
                )
                .reshape(-1)
            )

            return predictions


        # ------------------------------------------
        # B. BACKGROUND DATA
        # ------------------------------------------

        # Use at most 50 training windows.
        rng = np.random.default_rng(42)

        background_size = min(
            50,
            len(X_train)
        )

        background_indices = rng.choice(
            len(X_train),
            size=background_size,
            replace=False
        )

        background = X_train[
            background_indices
        ]


        background_flat = background.reshape(
            background_size,
            -1
        )


        # Reduce the background to representative
        # points to keep Kernel SHAP manageable.

        kmeans_clusters = min(
            10,
            background_size
        )

        background_summary = shap.kmeans(
            background_flat,
            kmeans_clusters
        )


        # ------------------------------------------
        # C. TEST SAMPLE
        # ------------------------------------------

        # Use the first 10 test forecasts to keep
        # computation time reasonable.

        sample_size = min(
            10,
            len(X_test)
        )

        test_sample = X_test[
            :sample_size
        ]

        test_flat = test_sample.reshape(
            sample_size,
            -1
        )


        # ------------------------------------------
        # D. KERNEL EXPLAINER
        # ------------------------------------------

        explainer = shap.KernelExplainer(
            predict_flat,
            background_summary
        )


        raw_shap_values = explainer.shap_values(
            test_flat,
            nsamples=100
        )


        # ------------------------------------------
        # E. NORMALIZE SHAP OUTPUT
        # ------------------------------------------

        if isinstance(
            raw_shap_values,
            list
        ):

            raw_shap_values = (
                raw_shap_values[0]
            )


        shap_array = np.asarray(
            raw_shap_values,
            dtype=float
        )


        # Some SHAP versions return:
        # (samples, flattened_features, 1)

        if shap_array.ndim == 3:

            shap_array = np.squeeze(
                shap_array,
                axis=-1
            )


        expected_flat_features = (
            lookback * n_features
        )


        if shap_array.shape != (
            sample_size,
            expected_flat_features
        ):

            st.error(
                "Unexpected SHAP output shape: "
                f"{shap_array.shape}"
            )

            st.stop()


        shap_sequences = shap_array.reshape(
            sample_size,
            lookback,
            n_features
        )


        # ------------------------------------------
        # F. GLOBAL IMPORTANCE
        # ------------------------------------------

        global_importance = (
            np.abs(
                shap_sequences
            )
            .mean(
                axis=(0, 1)
            )
        )


        global_df = pd.DataFrame({
            "Feature": features,
            "Mean Absolute SHAP": global_importance,
        }).sort_values(
            "Mean Absolute SHAP",
            ascending=False
        ).reset_index(
            drop=True
        )


        top_feature = (
            global_df.iloc[0][
                "Feature"
            ]
        )


        top_feature_index = (
            features.index(
                top_feature
            )
        )


        # ------------------------------------------
        # G. ONE FORECAST CONTRIBUTIONS
        # ------------------------------------------

        # Sum contribution across all 12 months
        # for each feature.

        one_forecast_values = (
            shap_sequences[0]
            .sum(axis=0)
        )


        one_forecast_df = pd.DataFrame({
            "Feature": features,
            "SHAP Contribution": one_forecast_values,
        }).sort_values(
            "SHAP Contribution",
            key=np.abs,
            ascending=False
        ).reset_index(
            drop=True
        )


        # ------------------------------------------
        # H. DEPENDENCE DATA
        # ------------------------------------------

        # Use the most recent month in each input
        # sequence for the top feature.

        dependence_values = (
            test_sample[
                :,
                -1,
                top_feature_index
            ]
        )


        dependence_shap = (
            shap_sequences[
                :,
                -1,
                top_feature_index
            ]
        )


        dependence_df = pd.DataFrame({
            "Feature Value": dependence_values,
            "SHAP Value": dependence_shap,
        })


        # ------------------------------------------
        # I. SAVE RESULTS
        # ------------------------------------------

        st.session_state.explain_report = {
            "global_importance":
                global_df,

            "one_forecast":
                one_forecast_df,

            "top_feature":
                top_feature,

            "dependence":
                dependence_df,

            "sample_size":
                sample_size,
        }


        st.success(
            "SHAP explanations completed!"
        )


# --------------------------------------------------
# 6. DISPLAY RESULTS
# --------------------------------------------------

if "explain_report" in st.session_state:

    report = (
        st.session_state.explain_report
    )


    # ----------------------------------------------
    # GLOBAL FEATURE IMPORTANCE
    # ----------------------------------------------

    st.subheader(
        "3. Global Feature Importance"
    )

    st.write(
        "Mean absolute SHAP values show how strongly "
        "each predictor influenced the sampled forecasts."
    )


    global_chart = (
        report[
            "global_importance"
        ]
        .set_index(
            "Feature"
        )
    )


    st.bar_chart(
        global_chart
    )


    st.dataframe(
        report[
            "global_importance"
        ],
        use_container_width=True
    )


    # ----------------------------------------------
    # ONE FORECAST
    # ----------------------------------------------

    st.subheader(
        "4. One Forecast — Feature Contributions"
    )

    st.write(
        "Positive SHAP values push the prediction upward, "
        "while negative values push it downward relative "
        "to the explainer's baseline."
    )


    one_chart = (
        report[
            "one_forecast"
        ]
        .set_index(
            "Feature"
        )
    )


    st.bar_chart(
        one_chart
    )


    st.dataframe(
        report[
            "one_forecast"
        ],
        use_container_width=True
    )


    # ----------------------------------------------
    # DEPENDENCE PLOT
    # ----------------------------------------------

    st.subheader(
        f"5. Dependence Plot — "
        f"{report['top_feature']}"
    )


    st.write(
        "This plot shows the relationship learned by "
        "the model for the most influential predictor. "
        "It should not be interpreted as proof of causation."
    )


    st.scatter_chart(
        report[
            "dependence"
        ],
        x="Feature Value",
        y="SHAP Value"
    )


    st.success(
        f"SHAP analysis used "
        f"{report['sample_size']} test forecasts."
    )


else:

    st.info(
        "Click Compute SHAP Explanations to begin."
    )

page_footer(7)
