import pandas as pd
import streamlit as st
from ui import page_header, page_footer




# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------


page_header(2)


st.write(
    "Check duplicate monthly records, identify missing "
    "values, and detect unusually high or low tourist "
    "arrivals using the IQR method."
)




# --------------------------------------------------
# CHECK PREVIOUS PAGE
# --------------------------------------------------


if "raw_df" not in st.session_state:


    st.warning(
        "Please open the Dataset page first."
    )


    st.stop()




# --------------------------------------------------
# CLEANING FUNCTION
# --------------------------------------------------


def clean_dataset(raw_df):

    # Preserve the original dataset.
    df = raw_df.copy()

    # 1. Remove exact duplicate rows.
    exact_duplicates = int(df.duplicated().sum())

    df = df.drop_duplicates(keep="first").copy()

    # 2. Find conflicting records for the same month.
    duplicate_dates = df[
        df.duplicated(subset=["date"], keep=False)
    ].copy()

    duplicate_month_count = int(
        duplicate_dates["date"].nunique()
    )

    # Exclude unresolved duplicate months.
    # We cannot determine the correct June 2006 arrivals.
    conflicting_months = duplicate_dates["date"].unique()

    if len(conflicting_months) > 0:
        df = df[
            ~df["date"].isin(conflicting_months)
        ].copy()

    # 3. Record missing values before excluding targets.
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    total_missing = int(df.isna().sum().sum())

    missing_rows = df[
        df.isna().any(axis=1)
    ].copy()

    # 4. Identify missing tourist arrivals.
    missing_target_rows = df[
        df["arrivals"].isna()
    ].copy()

    missing_target_count = len(missing_target_rows)

    # Exclude missing targets without inventing values.
    df = df.dropna(subset=["arrivals"]).copy()

    # 5. Calculate IQR outlier limits.
    q1 = df["arrivals"].quantile(0.25)
    q3 = df["arrivals"].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (df["arrivals"] < lower_bound)
        | (df["arrivals"] > upper_bound)
    )

    flagged_outliers = df.loc[
        outlier_mask,
        ["date", "arrivals"]
    ].copy()

    # 6. Sort the cleaned data.
    df = df.sort_values("date").reset_index(drop=True)

    # 7. Save the cleaning report.
    report = {
        "exact_duplicates": exact_duplicates,
        "duplicate_month_count": duplicate_month_count,
        "duplicate_dates": duplicate_dates,
        "missing_counts": missing_counts,
        "total_missing": total_missing,
        "missing_rows": missing_rows,
        "missing_target_rows": missing_target_rows,
        "missing_target_count": missing_target_count,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "flagged_outliers": flagged_outliers,
        "outlier_count": len(flagged_outliers),
        "rows_after_cleaning": len(df),
    }

    return df, report




# --------------------------------------------------
# RUN CLEANING
# --------------------------------------------------


if st.button("Run Cleaning", type="primary"):


    # Clear dependent results if cleaning is rerun.
    downstream_keys = [
        "selected_features",
        "feature_report",
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


    cleaned_df, report = clean_dataset(
        st.session_state.raw_df
    )


    # Save results for later pages.
    st.session_state.clean_df = cleaned_df


    st.session_state.clean_report = report


    st.success(
        "Cleaning checks completed successfully."
    )




# --------------------------------------------------
# DISPLAY STORED RESULTS
# --------------------------------------------------


if "clean_report" not in st.session_state:


    st.info(
        "Click 'Run Cleaning' to inspect the dataset."
    )


    st.stop()




report = st.session_state.clean_report


df = st.session_state.clean_df




# --------------------------------------------------
# 1. DUPLICATE RECORDS
# --------------------------------------------------


st.subheader("1. Duplicate Records")


col1, col2 = st.columns(2)


with col1:


    st.metric(
        "Exact Duplicates Removed",
        report["exact_duplicates"]
    )


with col2:


    st.metric(
        "Duplicate Monthly Dates",
        report["duplicate_month_count"]
    )




if report["duplicate_month_count"] > 0:


    st.error(
        "Conflicting records may exist for the same "
        "month. Review these records before proceeding."
    )


    st.dataframe(
        report["duplicate_dates"],
        use_container_width=True
    )


else:


    st.success(
        "No duplicate monthly dates remain."
    )




# --------------------------------------------------
# 2. MISSING VALUES
# --------------------------------------------------


st.subheader("2. Missing Values")


st.metric(
    "Total Missing Values",
    report["total_missing"]
)




if report["total_missing"] > 0:


    st.warning(
        "Missing values were detected. "
        "Review the affected columns and dates."
    )


    missing_table = (
        report["missing_counts"]
        .rename("Missing Count")
        .reset_index()
    )


    missing_table.columns = [
        "Column",
        "Missing Count"
    ]


    st.write("Missing values by column:")


    st.dataframe(
        missing_table,
        use_container_width=True
    )


    st.write("Rows containing missing values:")


    st.dataframe(
        report["missing_rows"],
        use_container_width=True
    )


else:


    st.success(
        "No missing values detected."
    )




# --------------------------------------------------
# 3. IQR OUTLIER CHECK
# --------------------------------------------------


st.subheader("3. IQR Outlier Detection")


col1, col2, col3 = st.columns(3)


with col1:


    st.metric(
        "Q1",
        f"{report['q1']:,.2f}"
    )


with col2:


    st.metric(
        "Q3",
        f"{report['q3']:,.2f}"
    )


with col3:


    st.metric(
        "IQR",
        f"{report['iqr']:,.2f}"
    )




st.write(
    f"Lower bound: {report['lower_bound']:,.2f}"
)


st.write(
    f"Upper bound: {report['upper_bound']:,.2f}"
)


st.metric(
    "Flagged Outliers",
    report["outlier_count"]
)




if report["outlier_count"] > 0:


    st.warning(
        "The following observations fall outside "
        "the IQR limits. They have NOT been removed."
    )


    st.dataframe(
        report["flagged_outliers"],
        use_container_width=True
    )


else:


    st.success(
        "No tourist arrivals outliers detected."
    )




# --------------------------------------------------
# 4. CLEANED DATASET PREVIEW
# --------------------------------------------------


st.subheader("4. Dataset After Cleaning Checks")


st.write(
    f"Rows remaining: {report['rows_after_cleaning']}"
)


st.dataframe(
    df.head(),
    use_container_width=True
)




# --------------------------------------------------
# 5. FINAL STATUS
# --------------------------------------------------


if report["duplicate_month_count"] > 0:
    st.warning(
        "Conflicting monthly records have been excluded "
        "from the modeling dataset."
    )

if report["missing_target_count"] > 0:
    st.warning(
        f"{report['missing_target_count']} months with "
        "missing arrivals have been excluded."
    )

remaining_missing = int(df.isna().sum().sum())

if remaining_missing > 0:
    st.info(
        f"{remaining_missing} missing predictor values remain. "
        "These will be handled during model preparation."
    )

st.success(
    "Cleaning completed. The original dataset is preserved."
)

page_footer(2)
