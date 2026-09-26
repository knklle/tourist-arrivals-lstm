import numpy as np
import pandas as pd
import streamlit as st
from ui import page_header, page_footer

from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor,
)


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

page_header(3)

st.write(
    "Select predictors for Philippine tourist arrivals "
    "using Spearman correlation and iterative VIF."
)


# --------------------------------------------------
# CHECK PREVIOUS STEP
# --------------------------------------------------

if "clean_df" not in st.session_state:

    st.warning(
        "Cleaned data is not available. "
        "Please open the Clean page and click Run Cleaning."
    )

    st.stop()


# --------------------------------------------------
# LOAD CLEANED DATA
# --------------------------------------------------

df = st.session_state.clean_df.copy()

df = df.sort_values(
    "date"
).reset_index(drop=True)


# --------------------------------------------------
# CANDIDATE PREDICTORS
# --------------------------------------------------

CANDIDATES = [
    "quarter",
    "is_holiday_peak",
    "temp_mean_c",
    "temp_min_c",
    "temp_max_c",
    "rainfall_mm",
    "rainy_days",
    "humidity_pct",
    "typhoon_count",
    "typhoon_max_wind_kt",
    "storm_signal_days",
    "pm25_ugm3",
    "wave_height_m",
]


# --------------------------------------------------
# VALIDATE REQUIRED COLUMNS
# --------------------------------------------------

required_columns = [
    "date",
    "arrivals",
] + CANDIDATES

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"Missing required columns: {missing_columns}"
    )

    st.stop()


# --------------------------------------------------
# TRAINING / TEST SPLIT
# --------------------------------------------------

st.subheader("1. Chronological Data Split")

train_ratio = st.slider(
    "Training split ratio",
    min_value=0.60,
    max_value=0.90,
    value=0.80,
    step=0.05,
)

split_idx = int(
    len(df) * train_ratio
)

train_df = df.iloc[
    :split_idx
].copy()

test_df = df.iloc[
    split_idx:
].copy()


col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Training Rows",
        len(train_df)
    )

with col2:

    st.metric(
        "Test Rows",
        len(test_df)
    )


st.write(
    "Training period:",
    train_df["date"].min().strftime("%Y-%m"),
    "to",
    train_df["date"].max().strftime("%Y-%m")
)

st.write(
    "Test period:",
    test_df["date"].min().strftime("%Y-%m"),
    "to",
    test_df["date"].max().strftime("%Y-%m")
)


st.info(
    "Feature selection uses training-period observations "
    "only. The test period is not used to select predictors."
)


# --------------------------------------------------
# SPEARMAN + VIF FUNCTION
# --------------------------------------------------

def select_features(training_data):

    X = training_data[CANDIDATES].copy()

    y = training_data["arrivals"].copy()

    # Convert numeric predictors safely.
    for col in CANDIDATES:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    y = pd.to_numeric(
        y,
        errors="coerce"
    )

    # ----------------------------------------------
    # 1. TRAINING-ONLY MISSING VALUE HANDLING
    # ----------------------------------------------

    # Use training-period medians only.
    # No test observations are used here.

    medians = X.median()

    X = X.fillna(medians)

    # Exclude missing target values from the
    # correlation calculation.

    valid_target = y.notna()

    X = X.loc[valid_target].copy()

    y = y.loc[valid_target].copy()

    # ----------------------------------------------
    # 2. SPEARMAN CORRELATION
    # ----------------------------------------------

    results = []

    for col in CANDIDATES:

        if X[col].nunique() < 2:

            rho = np.nan

            p_value = np.nan

        else:

            rho, p_value = spearmanr(
                X[col],
                y
            )

        keep = (
            np.isfinite(rho)
            and np.isfinite(p_value)
            and abs(rho) > 0.10
            and p_value < 0.05
        )

        results.append({
            "Feature": col,
            "Spearman rho": rho,
            "p-value": p_value,
            "Passes Filter": keep,
        })

    spearman_table = pd.DataFrame(results)

    kept_features = spearman_table.loc[
        spearman_table["Passes Filter"],
        "Feature"
    ].tolist()

    if not kept_features:

        return (
            spearman_table,
            pd.DataFrame(),
            [],
            medians
        )

    # ----------------------------------------------
    # 3. ITERATIVE VIF
    # ----------------------------------------------

    X_vif = X[kept_features].copy()

    vif_log = []

    while X_vif.shape[1] > 1:

        # Standardize using training-period data.
        # This helps numerical stability.

        means = X_vif.mean()

        stds = X_vif.std(ddof=0)

        # A constant column cannot contribute
        # independent variation.

        constant_cols = stds[
            (stds == 0) | stds.isna()
        ].index.tolist()

        if constant_cols:

            drop_col = constant_cols[0]

            vif_log.append({
                "Dropped Feature": drop_col,
                "VIF": np.inf,
                "Reason": "Constant predictor",
            })

            X_vif = X_vif.drop(
                columns=[drop_col]
            )

            continue

        Z = (X_vif - means) / stds

        # Compute VIF with an intercept.
        # Standardized columns have mean zero.

        vifs = []

        for i in range(Z.shape[1]):

            try:

                value = variance_inflation_factor(
                    Z.to_numpy(dtype=float),
                    i
                )

            except Exception:

                value = np.inf

            if not np.isfinite(value):

                value = np.inf

            vifs.append(float(value))

        max_vif = max(vifs)

        if max_vif < 5:

            break

        drop_idx = int(
            np.argmax(vifs)
        )

        drop_col = X_vif.columns[drop_idx]

        vif_log.append({
            "Dropped Feature": drop_col,
            "VIF": max_vif,
            "Reason": "VIF >= 5",
        })

        X_vif = X_vif.drop(
            columns=[drop_col]
        )

    # ----------------------------------------------
    # 4. FINAL SELECTED FEATURES
    # ----------------------------------------------

    selected_features = list(
        X_vif.columns
    )

    vif_table = pd.DataFrame(vif_log)

    return (
        spearman_table,
        vif_table,
        selected_features,
        medians
    )


# --------------------------------------------------
# RUN FEATURE SELECTION
# --------------------------------------------------

st.subheader("2. Spearman Correlation and VIF")

if st.button(
    "Run Spearman + VIF",
    type="primary"
):

    with st.spinner(
        "Calculating correlations and VIF..."
    ):

        (
            spearman_table,
            vif_table,
            selected_features,
            medians
        ) = select_features(train_df)

    # Clear old model results when features change.
    downstream_keys = [
        "scaler_X",
        "scaler_y",
        "X_train_seq",
        "y_train_seq",
        "X_test_seq",
        "y_test_seq",
        "test_df",
        "prepare_report",
        "model",
        "train_report",
        "evaluate_report",
        "explain_report",
        "forecast_report",
    ]

    for key in downstream_keys:

        st.session_state.pop(key, None)

    # Save feature-selection results.
    st.session_state.selected_features = (
        selected_features
    )

    st.session_state.feature_report = {
        "spearman": spearman_table,
        "vif": vif_table,
        "selected_features": selected_features,
        "train_ratio": train_ratio,
        "split_idx": split_idx,
        "medians": medians,
    }

    st.success(
        "Feature selection completed."
    )


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if "feature_report" in st.session_state:

    report = st.session_state.feature_report

    # Avoid displaying results from an old split.
    if report["split_idx"] != split_idx:

        st.warning(
            "The training split has changed. "
            "Click Run Spearman + VIF again."
        )

        st.stop()

    st.subheader("3. Spearman Correlation Results")

    st.dataframe(
        report["spearman"],
        use_container_width=True
    )

    st.subheader("4. VIF Removal Log")

    if not report["vif"].empty:

        st.dataframe(
            report["vif"],
            use_container_width=True
        )

    else:

        st.info(
            "No predictors were removed by VIF."
        )

    st.subheader("5. Final Selected Features")

    if report["selected_features"]:

        st.success(
            f"Selected {len(report['selected_features'])} "
            "predictors."
        )

        st.write(
            report["selected_features"]
        )

    else:

        st.error(
            "No features passed the selection criteria. "
            "Review the correlation results."
        )

else:

    st.info(
        "Click Run Spearman + VIF to begin "
        "feature selection."
    )

page_footer(3)
