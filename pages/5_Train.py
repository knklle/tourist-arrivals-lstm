import itertools
import random

import numpy as np
import pandas as pd
import streamlit as st
from ui import page_header, page_footer


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

page_header(5)

st.write(
    "Train an LSTM model using the prepared sequences "
    "and choose hyperparameters using validation data "
    "from the training period only."
)


# --------------------------------------------------
# 1. CHECK PREVIOUS STEP
# --------------------------------------------------

required_keys = [
    "X_train_seq",
    "y_train_seq",
    "X_test_seq",
    "y_test_seq",
    "selected_features",
    "prepare_report",
]

missing_keys = [
    key for key in required_keys
    if key not in st.session_state
]

if missing_keys:
    st.warning(
        "Prepared data is not available. "
        "Please complete the Prepare page first."
    )
    st.stop()


# --------------------------------------------------
# 2. LOAD PREPARED TRAINING DATA
# --------------------------------------------------

X_all = np.asarray(
    st.session_state.X_train_seq,
    dtype=np.float32
)

y_all = np.asarray(
    st.session_state.y_train_seq,
    dtype=np.float32
)

lookback = X_all.shape[1]
n_features = X_all.shape[2]


# --------------------------------------------------
# 3. DISPLAY INPUT INFORMATION
# --------------------------------------------------

st.subheader("1. Prepared Training Data")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Training Windows",
        len(X_all)
    )

with col2:
    st.metric(
        "Lookback",
        f"{lookback} months"
    )

with col3:
    st.metric(
        "Predictors",
        n_features
    )

st.write(
    "Training sequence shape:",
    X_all.shape
)

st.write(
    "Training target shape:",
    y_all.shape
)


# --------------------------------------------------
# 4. CHRONOLOGICAL VALIDATION SPLIT
# --------------------------------------------------

VALIDATION_RATIO = 0.15

validation_size = max(
    1,
    int(len(X_all) * VALIDATION_RATIO)
)

training_size = (
    len(X_all) - validation_size
)

if training_size < 1:
    st.error(
        "Not enough training sequences to create "
        "a validation period."
    )
    st.stop()


# Earlier windows = fitting.
# Latest training windows = validation.

X_train = X_all[:training_size]
y_train = y_all[:training_size]

X_val = X_all[training_size:]
y_val = y_all[training_size:]


st.subheader("2. Training / Validation Split")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Model-Fitting Windows",
        len(X_train)
    )

with col2:
    st.metric(
        "Validation Windows",
        len(X_val)
    )

st.info(
    "The validation set uses the latest 15% of the "
    "training sequences. The held-out test set is not "
    "used during hyperparameter tuning."
)


# --------------------------------------------------
# 5. HYPERPARAMETER GRID
# --------------------------------------------------

PARAM_GRID = {
    "units": [32, 64],
    "dropout": [0.1, 0.3],
    "batch_size": [16, 32],
}

combinations = list(
    itertools.product(
        PARAM_GRID["units"],
        PARAM_GRID["dropout"],
        PARAM_GRID["batch_size"],
    )
)

st.subheader("3. Hyperparameter Search")

st.write(
    f"Parameter combinations to test: "
    f"{len(combinations)}"
)

st.write(
    "LSTM units:",
    PARAM_GRID["units"]
)

st.write(
    "Dropout:",
    PARAM_GRID["dropout"]
)

st.write(
    "Batch size:",
    PARAM_GRID["batch_size"]
)


# --------------------------------------------------
# 6. REPRODUCIBILITY
# --------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------
# 7. MODEL BUILDER
# --------------------------------------------------

def build_model(
    units,
    dropout,
    lookback,
    n_features
):
    from tensorflow.keras.layers import (
        Input,
        LSTM,
        Dropout,
        Dense,
    )

    from tensorflow.keras.models import Sequential

    model = Sequential([
        Input(
            shape=(lookback, n_features)
        ),

        LSTM(
            units=units
        ),

        Dropout(
            rate=dropout
        ),

        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
    )

    return model


# --------------------------------------------------
# 8. TRAIN AND TUNE
# --------------------------------------------------

if st.button(
    "Train & Tune LSTM",
    type="primary"
):

    import tensorflow as tf

    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import EarlyStopping

    tf.random.set_seed(SEED)


    # Remove old results if training is rerun.
    for key in [
        "model",
        "train_report",
        "evaluate_report",
        "explain_report",
        "forecast_report",
    ]:

        st.session_state.pop(
            key,
            None
        )


    best_val_loss = float("inf")

    best_params = None
    best_weights = None
    best_history = None

    search_results = []


    progress = st.progress(0)

    status = st.empty()


    for run_number, (
        units,
        dropout,
        batch_size
    ) in enumerate(
        combinations,
        start=1
    ):

        status.write(
            f"Training model {run_number} of "
            f"{len(combinations)} — "
            f"units={units}, "
            f"dropout={dropout}, "
            f"batch size={batch_size}"
        )


        # Clear previous Keras model.
        K.clear_session()

        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)


        candidate = build_model(
            units=units,
            dropout=dropout,
            lookback=lookback,
            n_features=n_features,
        )


        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            mode="min",
        )


        history = candidate.fit(
            X_train,
            y_train,

            validation_data=(
                X_val,
                y_val
            ),

            epochs=100,

            batch_size=batch_size,

            shuffle=False,

            callbacks=[
                early_stopping
            ],

            verbose=0,
        )


        val_loss = float(
            min(
                history.history[
                    "val_loss"
                ]
            )
        )


        epochs_run = len(
            history.history[
                "loss"
            ]
        )


        search_results.append({
            "Units": units,
            "Dropout": dropout,
            "Batch Size": batch_size,
            "Best Validation Loss": val_loss,
            "Epochs Run": epochs_run,
        })


        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_params = {
                "units": units,
                "dropout": dropout,
                "batch_size": batch_size,
            }


            best_weights = [
                np.array(
                    weight,
                    copy=True
                )
                for weight
                in candidate.get_weights()
            ]


            best_history = {
                "loss": list(
                    history.history[
                        "loss"
                    ]
                ),

                "val_loss": list(
                    history.history[
                        "val_loss"
                    ]
                ),
            }


        progress.progress(
            run_number / len(combinations)
        )


    # --------------------------------------------------
    # 9. REBUILD BEST MODEL
    # --------------------------------------------------

    K.clear_session()


    best_model = build_model(
        units=best_params["units"],
        dropout=best_params["dropout"],
        lookback=lookback,
        n_features=n_features,
    )


    best_model.set_weights(
        best_weights
    )


    # --------------------------------------------------
    # 10. STORE RESULTS
    # --------------------------------------------------

    st.session_state.model = (
        best_model
    )


    st.session_state.train_report = {

        "best_params":
            best_params,

        "best_val_loss":
            best_val_loss,

        "loss_history":
            best_history["loss"],

        "val_loss_history":
            best_history["val_loss"],

        "search_results":
            search_results,

        "training_windows":
            len(X_train),

        "validation_windows":
            len(X_val),
    }


    status.empty()

    progress.progress(1.0)


    st.success(
        "LSTM training and tuning completed!"
    )


# --------------------------------------------------
# 11. DISPLAY RESULTS
# --------------------------------------------------

if "train_report" in st.session_state:

    report = (
        st.session_state.train_report
    )


    st.subheader(
        "4. Hyperparameter Search Results"
    )


    search_df = pd.DataFrame(
        report["search_results"]
    )


    st.dataframe(
        search_df,
        use_container_width=True
    )


    st.subheader(
        "5. Best Hyperparameters"
    )


    best = report[
        "best_params"
    ]


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "LSTM Units",
            best["units"]
        )


    with col2:

        st.metric(
            "Dropout",
            best["dropout"]
        )


    with col3:

        st.metric(
            "Batch Size",
            best["batch_size"]
        )


    st.write(
        "Best validation loss:",
        f"{report['best_val_loss']:.6f}"
    )


    st.write(
        "Model-fitting windows:",
        report["training_windows"]
    )


    st.write(
        "Validation windows:",
        report["validation_windows"]
    )


    # --------------------------------------------------
    # TRAINING CURVE
    # --------------------------------------------------

    st.subheader(
        "6. Training and Validation Loss"
    )


    loss_df = pd.DataFrame({

        "Training Loss":
            report["loss_history"],

        "Validation Loss":
            report["val_loss_history"],
    })


    st.line_chart(
        loss_df
    )


    st.success(
        "The best LSTM model is saved and ready "
        "for evaluation on the held-out test set."
    )


else:

    st.info(
        "Click Train & Tune LSTM to begin training."
    )

page_footer(5)
