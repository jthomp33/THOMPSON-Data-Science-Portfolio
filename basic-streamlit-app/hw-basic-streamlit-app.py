#This is a basic streamlit app using the penguins csv file. 


import streamlit as st
import pandas as pd

# Load the dataset

df = pd.read_csv("data/penguins.csv")  # Adjust column names if needed

data = df

# Title and description
st.title("Comprehensive Database")
st.write("This app allows you to explore the Palmer Penguins dataset with interactive filtering options by species.")

# Show a sample of the dataset
st.subheader("Sample Data")
st.dataframe(data.head())

# Interactive filtering options
species = st.selectbox("Select species:", options=["All"] + list(data["species"].unique()))

# Filter data based on selection
filtered_data = data if species == "All" else data[data["species"] == species]

# Slider for filtering by body mass
min_mass, max_mass = int(data["body_mass_g"].min()), int(data["body_mass_g"].max())
body_mass = st.slider("Select body mass range (g):", min_value=min_mass, max_value=max_mass, value=(min_mass, max_mass))
filtered_data = filtered_data[(filtered_data["body_mass_g"] >= body_mass[0]) & (filtered_data["body_mass_g"] <= body_mass[1])]

# Slider for filtering by flipper length
min_flipper, max_flipper = int(data["flipper_length_mm"].min()), int(data["flipper_length_mm"].max())
flipper_length = st.slider("Select flipper length range (mm):", min_value=min_flipper, max_value=max_flipper, value=(min_flipper, max_flipper))
filtered_data = filtered_data[(filtered_data["flipper_length_mm"] >= flipper_length[0]) & (filtered_data["flipper_length_mm"] <= flipper_length[1])]


# Display filtered data
st.subheader(f"There are {len(filtered_data)} penguins that match your specifications")
st.dataframe(filtered_data)

