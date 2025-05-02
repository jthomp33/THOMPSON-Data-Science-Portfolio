import pandas as pd            # Library for data manipulation
import seaborn as sns          # Library for statistical plotting
import matplotlib.pyplot as plt  # For creating custom plots
import streamlit as st         # Framework for building interactive web apps

# ================================================================================
# Missing Data & Data Quality Checks
#
# This lecture covers:
# - Data Validation: Checking data types, missing values, and ensuring consistency.
# - Missing Data Handling: Options to drop or impute missing data.
# - Visualization: Using heatmaps and histograms to explore data distribution.
# ================================================================================
st.title("Missing Data & Data Quality Checks")
st.markdown("""
This lecture covers:
- **Data Validation:** Checking data types, missing values, and basic consistency.
- **Missing Data Handling:** Options to drop or impute missing data.
- **Visualization:** Using heatmaps and histograms to understand data distribution.
""")

# ------------------------------------------------------------------------------
# Load the Dataset
# ------------------------------------------------------------------------------
# Read the Titanic dataset from a CSV file.
df = pd.read_csv("titanic.csv")

# ------------------------------------------------------------------------------
# Display Summary Statistics
# ------------------------------------------------------------------------------
# Show key statistical measures like mean, standard deviation, etc.
st.write("**Summary Statistics**")
st.dataframe(df.describe())

# ------------------------------------------------------------------------------
# Check for Missing Values
# ------------------------------------------------------------------------------
# Display the count of missing values for each column.
st.write("**Number of Missing Values by Column**")
st.dataframe(df.isnull().sum())

# ------------------------------------------------------------------------------
# Visualize Missing Data
# ------------------------------------------------------------------------------
# Create a heatmap to visually indicate where missing values occur.
st.write("Heatmap of the Missing values")
fig, ax = plt.subplots()
sns.heatmap(df.isnull(), cmap="viridis", cbar=False)
st.pyplot(fig)

# ================================================================================
# Interactive Missing Data Handling
#
# Users can select a numeric column and choose a method to address missing values.
# Options include:
# - Keeping the data unchanged
# - Dropping rows with missing values
# - Dropping columns if more than 50% of the values are missing
# - Imputing missing values with mean, median, or zero
# ================================================================================
st.subheader("Handle Missing Data")

# Work on a copy of the DataFrame so the original data remains unchanged.
column = st.selectbox("Choose a column to fill", df.select_dtypes(include=['number']).columns)

# Display selected column's data
st.dataframe(df[column])

# Choose missing data handling method
method = st.radio(
    "Choose a method",
    [
        "Original DF",
        "Drop Rows",
        "Drop Columns (>50% Missing)",
        "Impute Mean",
        "Impute Median",
        "Impute Zero",
    ]
)

# Copy the original DataFrame for processing
df_clean = df.copy()

# Apply the selected method
if method == "Original DF":
    pass
elif method == "Drop Rows":
    df_clean = df_clean.dropna()
elif method == "Drop Columns (>50% Missing)":
    # Drop columns where more than 50% of the values are missing
    df_clean = df_clean.drop(columns=df_clean.columns[df_clean.isnull().mean() > 0.5])
elif method == "Impute Mean":
    # Replace missing values in the selected column with the column's mean
    if column in df_clean.columns:
        df_clean[column] = df_clean[column].fillna(df_clean[column].mean())
elif method == "Impute Median":
    # Replace missing values in the selected column with the column's median
    if column in df_clean.columns:
        df_clean[column] = df_clean[column].fillna(df_clean[column].median())
elif method == "Impute Zero":
    # Replace missing values in the selected column with zero
    if column in df_clean.columns:
        df_clean[column] = df_clean[column].fillna(0)

# ------------------------------------------------------------------------------
# Compare Data Distributions: Original vs. Cleaned
fig,ax=plt.subplots()
sns.histplot(df_clean[column], kde= True)
st.pyplot(fig)
# ------------------------------------------------------------------------------
# Display the cleaned DataFrame
st.write("**Cleaned Data**")
st.dataframe(df_clean)
st.write(df_clean.describe())