
import numpy as np
import pandas as pd
import streamlit as st
from ui import page_header, page_footer

from sklearn.preprocessing import MinMaxScaler


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

page_header(4)

st.write(
    "Prepare the selected predictors for LSTM training "
    "using chronological splitting, training-only scaling, "
    "and 12-month input sequences."
)


# --------------------------------------------------
# 1. CHECK PREVIOUS STEPS
# --------------------------------------------------

required_keys = [
    "clean_df",
    "selected_features",
    "feature_report",
]

missing_keys = [
    key for key in required_keys
    if key not in st.session_state
]

if missing_keys:

    st.warning(
        "Please complete the Dataset, Clean, and "
        "Features pages before continuing."
    )

    st.stop()


# --------------------------------------------------
# 2. LOAD CLEANED DATA
# --------------------------------------------------

df = st.session_state.clean_df.copy()

df = df.sort_values(
    "date"
).reset_index(drop=True)

features = list(
    st.session_state.selected_features
)

feature_report = st.session_state.feature_report

LOOKBACK = 12


if not features:

    st.error(
        "No predictors were selected. "
        "Please rerun the Features page."
    )

    st.stop()


# --------------------------------------------------
# 3. VALIDATE REQUIRED COLUMNS
# --------------------------------------------------

required_columns = [
    "date",
    "arrivals",
] + features

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"Missing columns: {missing_columns}"
    )

    st.stop()


# --------------------------------------------------
# 4. USE THE SAME SPLIT AS FEATURE SELECTION
# --------------------------------------------------

train_ratio = float(
    feature_report["train_ratio"]
)

split_idx = int(
    feature_report["split_idx"]
)

expected_split = int(
    len(df) * train_ratio
)

if split_idx != expected_split:

    st.error(
        "The cleaned dataset or training split has "
        "changed since feature selection. Please rerun "
        "the Features page before preparing sequences."
    )

    st.stop()


st.subheader("1. Chronological Split")

st.write(
    f"Training ratio: {train_ratio:.0%}"
)

st.write(
    f"Testing ratio: {1 - train_ratio:.0%}"
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
    "Testing period:",
    test_df["date"].min().strftime("%Y-%m"),
    "to",
    test_df["date"].max().strftime("%Y-%m")
)


# --------------------------------------------------
# 5. DISPLAY SELECTED FEATURES
# --------------------------------------------------

st.subheader("2. Selected Predictors")

st.write(
    f"Number of predictors: {len(features)}"
)

st.write(features)


# --------------------------------------------------
# 6. SEQUENCE CREATION FUNCTION
# --------------------------------------------------

def make_sequences(
    X,
    y,
    dates,
    lookback
):

    Xs = []
    ys = []

    target_indices = []
    target_dates = []

    skipped_windows = 0

    # Convert dates to consecutive month numbers.
    # This allows us to detect calendar gaps.

    dates = pd.to_datetime(
        pd.Series(dates)
    ).reset_index(drop=True)

    month_numbers = (
        dates.dt.year.to_numpy() * 12
        + dates.dt.month.to_numpy()
    )

    # Each target must have 12 preceding months.
    for i in range(lookback, len(X)):

        start = i - lookback

        # Include the target month in the date check.
        # This prevents skipped months from being
        # treated as consecutive observations.

        window_months = month_numbers[
            start:i + 1
        ]

        is_consecutive = np.all(
            np.diff(window_months) == 1
        )

        if not is_consecutive:

            skipped_windows += 1

            continue

        # Input: previous 12 months of predictors.
        Xs.append(
            X[start:i]
        )

        # Target: arrivals in the next month.
        ys.append(
            y[i]
        )

        # Store target position for evaluation.
        target_indices.append(i)

        target_dates.append(
            dates.iloc[i]
        )

    n_features = X.shape[1]

    # Use explicit shapes even if no windows exist.
    Xs = np.asarray(
        Xs,
        dtype=np.float32
    ).reshape(
        -1,
        lookback,
        n_features
    )

    ys = np.asarray(
        ys,
        dtype=np.float32
    ).reshape(-1, 1)

    return (
        Xs,
        ys,
        target_indices,
        target_dates,
        skipped_windows
    )


# --------------------------------------------------
# 7. RUN PREPARATION
# --------------------------------------------------

st.subheader("3. Prepare LSTM Sequences")

st.info(
    "Scalers and missing-value replacements will "
    "be fitted using training data only. Windows "
    "crossing missing calendar months will be skipped."
)


if st.button(
    "Run Preparation",
    type="primary"
):

    with st.spinner(
        "Preparing the LSTM dataset..."
    ):

        # ------------------------------------------
        # A. PREPARE NUMERIC PREDICTORS
        # ------------------------------------------

        train_X_df = train_df[
            features
        ].copy()

        test_X_df = test_df[
            features
        ].copy()

        # Ensure predictors are numeric.

        for col in features:

            train_X_df[col] = pd.to_numeric(
                train_X_df[col],
                errors="coerce"
            )

            test_X_df[col] = pd.to_numeric(
                test_X_df[col],
                errors="coerce"
            )

        # Replace infinite values with missing values.

        train_X_df = train_X_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        test_X_df = test_X_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # ------------------------------------------
        # B. TRAINING-ONLY MISSING-VALUE HANDLING
        # ------------------------------------------

        train_medians = train_X_df.median()

        # Check whether a selected feature has
        # no valid training observations.

        if train_medians.isna().any():

            st.error(
                "At least one selected feature has "
                "no valid training observations."
            )

            st.stop()

        # Fit replacement values on training data only.

        train_X_df = train_X_df.fillna(
            train_medians
        )

        test_X_df = test_X_df.fillna(
            train_medians
        )

        # ------------------------------------------
        # C. PREPARE TARGET VALUES
        # ------------------------------------------

        train_y_df = train_df[
            ["arrivals"]
        ].copy()

        test_y_df = test_df[
            ["arrivals"]
        ].copy()

        train_y_df["arrivals"] = pd.to_numeric(
            train_y_df["arrivals"],
            errors="coerce"
        )

        test_y_df["arrivals"] = pd.to_numeric(
            test_y_df["arrivals"],
            errors="coerce"
        )

        # Missing targets should already have
        # been excluded during cleaning.

        if (
            train_y_df.isna().any().any()
            or test_y_df.isna().any().any()
        ):

            st.error(
                "Missing target values remain. "
                "Please review the Clean page."
            )

            st.stop()

        # ------------------------------------------
        # D. FIT SCALERS ON TRAINING DATA ONLY
        # ------------------------------------------

        scaler_X = MinMaxScaler()

        scaler_y = MinMaxScaler()

        scaler_X.fit(
            train_X_df
        )

        scaler_y.fit(
            train_y_df
        )

        # ------------------------------------------
        # E. TRANSFORM TRAINING AND TEST DATA
        # ------------------------------------------

        train_X_scaled = scaler_X.transform(
            train_X_df
        )

        test_X_scaled = scaler_X.transform(
            test_X_df
        )

        train_y_scaled = scaler_y.transform(
            train_y_df
        )

        test_y_scaled = scaler_y.transform(
            test_y_df
        )

        # ------------------------------------------
        # F. CREATE TRAINING SEQUENCES
        # ------------------------------------------

        (
            X_train_seq,
            y_train_seq,
            train_target_indices,
            train_target_dates,
            skipped_train
        ) = make_sequences(
            train_X_scaled,
            train_y_scaled,
            train_df["date"],
            LOOKBACK
        )

        # ------------------------------------------
        # G. CREATE TEST SEQUENCES
        # ------------------------------------------

        (
            X_test_seq,
            y_test_seq,
            test_target_indices,
            test_target_dates,
            skipped_test
        ) = make_sequences(
            test_X_scaled,
            test_y_scaled,
            test_df["date"],
            LOOKBACK
        )

        # ------------------------------------------
        # H. VALIDATE SEQUENCES
        # ------------------------------------------

        if len(X_train_seq) == 0:

            st.error(
                "No valid training sequences were "
                "created. Review the calendar gaps."
            )

            st.stop()

        if len(X_test_seq) == 0:

            st.error(
                "No valid test sequences were "
                "created. Review the test period."
            )

            st.stop()

        # ------------------------------------------
        # I. CLEAR OLD MODEL RESULTS
        # ------------------------------------------

        downstream_keys = [
            "model",
            "train_report",
            "evaluate_report",
            "explain_report",
            "forecast_report",
        ]

        for key in downstream_keys:

            st.session_state.pop(
                key,
                None
            )

        # ------------------------------------------
        # J. SAVE PREPARED DATA
        # ------------------------------------------

        st.session_state.scaler_X = scaler_X

        st.session_state.scaler_y = scaler_y

        st.session_state.feature_medians = (
            train_medians
        )

        st.session_state.X_train_seq = (
            X_train_seq
        )

        st.session_state.y_train_seq = (
            y_train_seq
        )

        st.session_state.X_test_seq = (
            X_test_seq
        )

        st.session_state.y_test_seq = (
            y_test_seq
        )

        st.session_state.train_df = (
            train_df.copy()
        )

        st.session_state.test_df = (
            test_df.copy()
        )

        st.session_state.train_target_indices = (
            train_target_indices
        )

        st.session_state.test_target_indices = (
            test_target_indices
        )

        st.session_state.train_target_dates = (
            train_target_dates
        )

        st.session_state.test_target_dates = (
            test_target_dates
        )

        st.session_state.prepare_report = {

            "train_ratio": train_ratio,

            "train_rows": len(train_df),

            "test_rows": len(test_df),

            "train_windows": len(X_train_seq),

            "test_windows": len(X_test_seq),

            "skipped_train": skipped_train,

            "skipped_test": skipped_test,

            "lookback": LOOKBACK,

            "n_features": len(features),

            "train_shape": X_train_seq.shape,

            "test_shape": X_test_seq.shape,

        }

        st.success(
            "LSTM data preparation completed!"
        )


# --------------------------------------------------
# 8. DISPLAY PREPARATION RESULTS
# --------------------------------------------------

if "prepare_report" in st.session_state:

    report = st.session_state.prepare_report

    st.subheader("4. Preparation Results")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Training Windows",
            report["train_windows"]
        )

    with col2:

        st.metric(
            "Test Windows",
            report["test_windows"]
        )

    st.write(
        "Training sequence shape:",
        report["train_shape"]
    )

    st.write(
        "Test sequence shape:",
        report["test_shape"]
    )

    st.write(
        "Lookback:",
        report["lookback"],
        "months"
    )

    st.write(
        "Number of features:",
        report["n_features"]
    )

    st.write(
        "Training windows skipped due to calendar gaps:",
        report["skipped_train"]
    )

    st.write(
        "Test windows skipped due to calendar gaps:",
        report["skipped_test"]
    )

    st.success(
        "Prepared sequences are ready for LSTM training."
    )

else:

    st.info(
        "Click Run Preparation to prepare the LSTM data."
    )


page_footer(4)
