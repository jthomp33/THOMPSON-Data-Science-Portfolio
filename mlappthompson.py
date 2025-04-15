import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, roc_curve, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree # <-- Import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris, fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------
# Sidebar
# ----------------------------------

st.sidebar.title("🔧 Model Controls")

dataset_option = st.sidebar.selectbox(
    "Choose a sample dataset or upload your own:",
    ["Iris (Classification)", "California Housing (Regression)", "Upload CSV"]
)

uploaded_file = None
if dataset_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])

# ----------------------------------
# Home instructions
# ----------------------------------

st.title("🧠 Interactive Machine Learning Explorer")

st.markdown("""
Welcome to the **ML Explorer App**! 🚀
Here's how to use it:

1.  **Choose a dataset** from the sidebar — select one of our built-in samples or upload your own CSV.
2.  The app will **automatically detect** whether it's a regression or classification task (you can override if needed).
3.  Choose a **model and tune hyperparameters**.
4.  View training **performance metrics and visualizations**.

**Note**: Make sure your dataset includes a column named `"target"` (or choose it in the dropdown).
""")

# ----------------------------------
# Load Dataset
# ----------------------------------

def load_sample_data(option):
    if option == "Iris (Classification)":
        data = load_iris(as_frame=True)
        df = data.frame
        # Keep target numeric initially for easier class name mapping if needed
        # df['target'] = df['target'].astype(str) # We will handle mapping later
        task = "classification"
        # Store original class names for visualization
        target_names = data.target_names
    elif option == "California Housing (Regression)":
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        # Use the original target column name
        # df['target'] = df['MedHouseVal'] # Let user select target
        task = "regression"
        target_names = None # No class names for regression
    else:
        df = None
        task = None
        target_names = None
    return df, task, target_names

df = None
task = None
original_target_names = None # Store original names for built-in datasets

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Dataset Preview:", df.head())
        # Initial task guess (user can override)
        potential_target = df.columns[-1] # Guess last column is target
        if potential_target in df.columns:
            if pd.api.types.is_numeric_dtype(df[potential_target]):
                 # Could be regression or classification with numeric labels
                 if df[potential_target].nunique() < 15: # Heuristic: few unique numbers might be classes
                     task = "classification"
                 else:
                     task = "regression"
            else:
                task = "classification"
        else:
            st.warning("Could not automatically identify a 'target' column. Please select one below.")
            task = None # Force user selection

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        df = None # Ensure df is None if loading fails
else:
    df, task, original_target_names = load_sample_data(dataset_option)
    if df is not None:
        st.write("📊 Sample Dataset Preview:", df.head())

# ----------------------------------
# Proceed if data loaded
# ----------------------------------

if df is not None:

    st.subheader("📌 Data & Task Setup")

    # Allow user to override task type
    if task:
        task_options = ["classification", "regression"]
        selected_task_index = task_options.index(task)
        task = st.radio("Detected Task Type (Override if needed):", task_options, index=selected_task_index)
    else:
        task = st.radio("Select Task Type:", ["classification", "regression"])


    # Select Target Column
    default_target_index = 0
    if 'target' in df.columns:
        default_target_index = df.columns.get_loc('target')
    elif task == "regression" and 'MedHouseVal' in df.columns and dataset_option == "California Housing (Regression)":
         default_target_index = df.columns.get_loc('MedHouseVal')
    elif task == "classification" and 'target' in df.columns and dataset_option == "Iris (Classification)":
         default_target_index = df.columns.get_loc('target')
    elif len(df.columns) > 0:
        default_target_index = len(df.columns) - 1 # Default to last column if no better guess

    target_column = st.selectbox("Select the target variable:", df.columns, index=default_target_index)

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # --- Data Preprocessing ---
        st.subheader("⚙️ Preprocessing")

        # Handle non-numeric features
        feature_cols_to_encode = X.select_dtypes(include=['object', 'category']).columns
        if not feature_cols_to_encode.empty:
            st.write(f"Encoding non-numeric features: {', '.join(feature_cols_to_encode)}")
            for col in feature_cols_to_encode:
                 # Use LabelEncoder for simplicity here, consider OneHotEncoder for non-ordinal data
                 X[col] = LabelEncoder().fit_transform(X[col])
        else:
            st.write("No non-numeric features detected to encode.")

        # Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns) # Keep column names

        # Handle target variable for classification
        y_encoder = None
        class_names_for_plot = None
        if task == "classification":
            # Check if target needs encoding
            if not pd.api.types.is_numeric_dtype(y):
                st.write(f"Encoding target variable '{target_column}'...")
                y_encoder = LabelEncoder()
                y = y_encoder.fit_transform(y)
                class_names_for_plot = y_encoder.classes_.astype(str).tolist() # Get names from encoder
            else:
                 # Target is numeric, might be classes (0, 1, 2...) or could be regression target mistakenly chosen
                 unique_targets = sorted(y.unique())
                 if len(unique_targets) < 15: # Heuristic check if it looks like classes
                     st.write(f"Target variable '{target_column}' is numeric. Assuming labels: {unique_targets}")
                     class_names_for_plot = [str(cls) for cls in unique_targets] # Use unique values as names
                 else:
                     st.warning(f"Target variable '{target_column}' is numeric with many unique values. Ensure this is intended for classification.")
                     # Attempt to use unique values anyway, might be messy
                     class_names_for_plot = [str(cls) for cls in unique_targets]

            # Use original names if available (e.g., from Iris) and encoder wasn't needed
            if original_target_names is not None and y_encoder is None:
                class_names_for_plot = original_target_names.tolist()


        # Train-test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=(y if task == 'classification' and y.nunique() > 1 else None))
            st.write(f"Data split into {len(X_train)} training samples and {len(X_test)} test samples.")
        except ValueError as e:
             st.error(f"Error during train-test split: {e}. This might happen if the test size is too small or classes have too few members for stratification.")
             # Exit gracefully if split fails
             st.stop()


        # ----------------------------------
        # Model Selection
        # ----------------------------------
        st.sidebar.subheader("🧠 Model Selection")
        if task == "classification":
            model_type = st.sidebar.selectbox("Choose a classifier", ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"])
        else: # Regression
            model_type = st.sidebar.selectbox("Choose a regressor", ["Linear Regression", "Decision Tree"])

        # Hyperparameters
        st.sidebar.subheader("⚙️ Hyperparameters")
        model = None # Initialize model
        if model_type == "Logistic Regression":
            C = st.sidebar.slider("Inverse Regularization Strength (C)", 0.01, 10.0, 1.0, 0.01)
            model = LogisticRegression(C=C, max_iter=1000, random_state=42)
        elif model_type == "Decision Tree":
            max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
            min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
            min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, 1)
            criterion_options = ["gini", "entropy"] if task == "classification" else ["squared_error", "friedman_mse", "absolute_error"]
            criterion = st.sidebar.selectbox("Criterion", criterion_options)

            if task == "classification":
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    criterion=criterion,
                    random_state=42
                )
            else: # Regression
                 model = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    criterion=criterion,
                    random_state=42
                 )
        elif model_type == "K-Nearest Neighbors": # Only for classification in this setup
            n_neighbors = st.sidebar.slider("Number of Neighbors (k)", 1, 15, 5)
            weights = st.sidebar.selectbox("Weights", ["uniform", "distance"])
            metric = st.sidebar.selectbox("Metric", ["minkowski", "euclidean", "manhattan"])
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
        elif model_type == "Linear Regression": # Only for regression
            model = LinearRegression()

        # ----------------------------------
        # Train and Evaluate
        # ----------------------------------
        if model: # Proceed only if a model was successfully defined
            st.subheader("🚀 Training & Evaluation")
            try:
                # Train the model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.write(f"**{model_type} Model Results:**")

                if task == "classification":
                    st.write("✅ **Classification Metrics**")
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                    col1, col2 = st.columns(2)
                    col1.metric("Accuracy", f"{accuracy:.3f}")
                    col1.metric("Precision", f"{precision:.3f}")
                    col2.metric("Recall", f"{recall:.3f}")
                    col2.metric("F1 Score", f"{f1:.3f}")


                    # Confusion Matrix
                    st.subheader("📉 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                                xticklabels=class_names_for_plot or sorted(np.unique(y_test)), # Use class names if available
                                yticklabels=class_names_for_plot or sorted(np.unique(y_test)))
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)

                    # ROC Curve - only for binary or if probabilities are available
                    if hasattr(model, "predict_proba") and len(np.unique(y)) == 2:
                         st.subheader("📈 ROC Curve")
                         y_probs = model.predict_proba(X_test)[:, 1]
                         fpr, tpr, _ = roc_curve(y_test, y_probs)
                         auc = roc_auc_score(y_test, y_probs)
                         fig_roc, ax_roc = plt.subplots()
                         ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
                         ax_roc.plot([0, 1], [0, 1], 'k--', label="Random Chance")
                         ax_roc.set_xlabel("False Positive Rate")
                         ax_roc.set_ylabel("True Positive Rate")
                         ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve")
                         ax_roc.legend()
                         st.pyplot(fig_roc)
                    elif len(np.unique(y)) > 2:
                         st.write("ROC curve is typically shown for binary classification tasks.")


                else: # Regression
                    st.write("📈 **Regression Metrics**")
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)

                    col1, col2 = st.columns(2)
                    col1.metric("R² Score", f"{r2:.3f}")
                    col2.metric("Mean Squared Error (MSE)", f"{mse:.3f}")
                    col1.metric("Mean Absolute Error (MAE)", f"{mae:.3f}")

                    # Prediction Plot
                    st.subheader("📊 Predictions vs Actuals")
                    fig_pred, ax_pred = plt.subplots()
                    ax_pred.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', s=50)
                    ax_pred.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Ideal Fit")
                    ax_pred.set_xlabel("True Values")
                    ax_pred.set_ylabel("Predicted Values")
                    ax_pred.set_title("Actual vs. Predicted Values")
                    ax_pred.legend()
                    ax_pred.grid(True, linestyle='--', alpha=0.6)
                    st.pyplot(fig_pred)

                    # Residual Plot
                    st.subheader("📉 Residual Plot")
                    residuals = y_test - y_pred
                    fig_res, ax_res = plt.subplots()
                    ax_res.scatter(y_pred, residuals, alpha=0.6, edgecolors='w', s=50)
                    ax_res.axhline(0, color='r', linestyle='--', lw=2, label="Zero Error")
                    ax_res.set_xlabel("Predicted Values")
                    ax_res.set_ylabel("Residuals (Actual - Predicted)")
                    ax_res.set_title("Residuals vs. Predicted Values")
                    ax_res.legend()
                    ax_res.grid(True, linestyle='--', alpha=0.6)
                    st.pyplot(fig_res)

                # --- ADD DECISION TREE PLOT HERE ---
                if model_type == "Decision Tree":
                    st.subheader("🌳 Decision Tree Structure")
                    st.write("Visualizing the trained decision tree (may take a moment for deep trees).")
                    try:
                        fig_tree, ax_tree = plt.subplots(figsize=(20, 10)) # Adjust size as needed
                        plot_tree(model,
                                  feature_names=X.columns.tolist(), # Use feature names from X
                                  class_names= class_names_for_plot if task == "classification" else None, # Use stored class names for classification
                                  filled=True,
                                  rounded=True,
                                  impurity=True,
                                  ax=ax_tree,
                                  fontsize=10,
                                  max_depth=5) # Limit plot depth for readability if desired

                        st.pyplot(fig_tree)
                        st.caption("Note: Visualization might be truncated (`max_depth=5` shown) or simplified for clarity. The actual trained model uses the full depth set in hyperparameters.")
                    except Exception as e:
                        st.error(f"Could not generate decision tree plot: {e}")
                # --- END DECISION TREE PLOT ---

            except Exception as e:
                st.error(f"An error occurred during model training or evaluation: {e}")
        else:
             st.warning("Model could not be initialized. Please check settings.")

    else:
        st.warning("Please select a target variable to proceed.")

elif uploaded_file is None and dataset_option == "Upload CSV":
    st.info("Upload a CSV file using the sidebar to begin.")
# No need for an else here because the initial sample dataset selection handles the non-upload case