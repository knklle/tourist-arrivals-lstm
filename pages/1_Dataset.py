
from pathlib import Path

import pandas as pd
import streamlit as st
from ui import page_header, page_footer


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

page_header(1)

st.write(
    "Load and inspect the Philippine tourist arrivals "
    "dataset before cleaning and feature selection."
)


# --------------------------------------------------
# DATASET PATH
# --------------------------------------------------

# Resolve the path relative to the project directory,
# rather than relying on the terminal's working directory.

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "tourist_arrivals.csv"


# --------------------------------------------------
# EXPECTED DATASET COLUMNS
# --------------------------------------------------

EXPECTED_COLUMNS = [
    "date",
    "year",
    "month",
    "quarter",
    "season",
    "monsoon",
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
    "arrivals",
]


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_dataset(file_path):
    """
    Load the CSV without modifying the original data.

    Dates are converted to datetime, and observations
    are sorted chronologically.
    """

    df = pd.read_csv(file_path, skiprows=2)

    # Remove accidental whitespace in column names.
    df.columns = df.columns.str.strip()

    # Ensure the date column exists.
    if "date" not in df.columns:
        raise ValueError(
            "The dataset does not contain a 'date' column."
        )

    # Convert dates safely.
    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    # Sort chronologically.
    df = df.sort_values(
        by="date",
        na_position="last",
    ).reset_index(drop=True)

    return df


# --------------------------------------------------
# CHECK FILE EXISTENCE
# --------------------------------------------------

if not DATA_PATH.is_file():

    st.error(
        "Dataset not found. Please place "
        "'tourist_arrivals.csv' inside the data folder."
    )

    st.code(str(DATA_PATH))

    st.stop()


# --------------------------------------------------
# READ THE CSV
# --------------------------------------------------

try:

    df = load_dataset(str(DATA_PATH))

except Exception as error:

    st.error("Unable to load the dataset.")

    st.exception(error)

    st.stop()


# --------------------------------------------------
# VALIDATE THE DATASET SCHEMA
# --------------------------------------------------

missing_columns = [
    col for col in EXPECTED_COLUMNS
    if col not in df.columns
]

extra_columns = [
    col for col in df.columns
    if col not in EXPECTED_COLUMNS
]

if missing_columns:

    st.error(
        "The dataset is missing required columns."
    )

    st.write("Missing columns:", missing_columns)

    st.stop()


if extra_columns:

    st.warning(
        "Additional columns were found in the dataset."
    )

    st.write("Extra columns:", extra_columns)


if df.empty:

    st.error("The dataset contains no observations.")

    st.stop()


# --------------------------------------------------
# STORE DATASET IN SESSION STATE
# --------------------------------------------------

# Store a copy for subsequent pages.
# Do not clean or modify the raw dataset here.

st.session_state.raw_df = df.copy()


# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        label="Total Rows",
        value=len(df),
    )

with col2:

    st.metric(
        label="Total Columns",
        value=len(df.columns),
    )

with col3:

    st.metric(
        label="Missing Values",
        value=int(df.isna().sum().sum()),
    )


# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

st.subheader("First Five Observations")

st.dataframe(
    df.head(),
    use_container_width=True,
)


# --------------------------------------------------
# DATA TYPES
# --------------------------------------------------

with st.expander("Inspect column types and missing values"):
    st.subheader("Column Data Types")

    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
    })

    st.dataframe(
        dtype_df,
        use_container_width=True,
    )


    # --------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------

    st.subheader("Missing Values by Column")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isna().sum().values,
    })

    missing_df["Missing Percentage"] = (
        missing_df["Missing Count"] / len(df) * 100
    ).round(2)

    st.dataframe(
        missing_df,
        use_container_width=True,
    )


# --------------------------------------------------
# DATE VALIDATION
# --------------------------------------------------

st.subheader("Monthly Sequence Validation")

invalid_dates = int(df["date"].isna().sum())

if invalid_dates > 0:

    st.error(
        f"Found {invalid_dates} missing or invalid dates. "
        "Resolve these before preparing sequences."
    )

    st.stop()


# Check whether each date represents the first day
# of a calendar month.

non_month_start = int(
    (df["date"].dt.day != 1).sum()
)

if non_month_start > 0:

    st.error(
        f"Found {non_month_start} dates that are not "
        "the first day of a month."
    )

    st.stop()


# --------------------------------------------------
# DUPLICATE DATE CHECK
# --------------------------------------------------

duplicate_dates = int(
    df.duplicated(subset=["date"]).sum()
)

if duplicate_dates > 0:

    st.warning(
        f"Found {duplicate_dates} duplicate date records. "
        "These will be reviewed during cleaning."
    )

else:

    st.success("No duplicate monthly dates found.")


# --------------------------------------------------
# MISSING MONTH CHECK
# --------------------------------------------------

expected_months = pd.date_range(
    start=df["date"].min(),
    end=df["date"].max(),
    freq="MS",
)

actual_months = pd.DatetimeIndex(
    df["date"].drop_duplicates()
)

missing_months = expected_months.difference(
    actual_months
)

if len(missing_months) > 0:

    st.warning(
        f"Found {len(missing_months)} missing months."
    )

    missing_month_df = pd.DataFrame({
        "Missing Month": missing_months,
    })

    st.dataframe(
        missing_month_df,
        use_container_width=True,
    )

else:

    st.success(
        "No gaps found in the monthly sequence."
    )


# --------------------------------------------------
# DATE RANGE
# --------------------------------------------------

st.subheader("Dataset Date Range")

st.write(
    "Earliest month:",
    df["date"].min().strftime("%Y-%m-%d"),
)

st.write(
    "Latest month:",
    df["date"].max().strftime("%Y-%m-%d"),
)


# --------------------------------------------------
# TARGET VARIABLE PREVIEW
# --------------------------------------------------

st.subheader("Tourist Arrivals Over Time")

if pd.api.types.is_numeric_dtype(df["arrivals"]):

    chart_df = (
        df[["date", "arrivals"]]
        .groupby("date", as_index=False)["arrivals"]
        .mean()
        .set_index("date")
    )

    st.line_chart(chart_df)

else:

    st.warning(
        "The arrivals column is not numeric. "
        "Its values must be inspected before modeling."
    )


# --------------------------------------------------
# COMPLETION MESSAGE
# --------------------------------------------------

st.success(
    "Dataset summary completed. "
    "Proceed to the Clean page after reviewing "
    "any warnings above."
)

page_footer(1)
