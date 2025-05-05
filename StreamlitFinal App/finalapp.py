import streamlit as st  # Streamlit for interactive web apps
import pandas as pd     # Data manipulation
import numpy as np      # Numerical operations
from sklearn.cluster import KMeans, AgglomerativeClustering  # Clustering algorithms
from sklearn.decomposition import PCA  # Dimensionality reduction
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt  # Static plotting
import seaborn as sns  # For heatmaps
import plotly.express as px  # Interactive plotting
from io import StringIO
import base64  # For downloading results

# -------------------
# Helper Functions
# -------------------
def load_sample_data(name):
    if name == "Iris":
        from sklearn.datasets import load_iris
        data = load_iris(as_frame=True)
        return data.frame
    elif name == "Wine Quality":
        return pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
    else:
        return None

def run_kmeans(data, n_clusters):
    if n_clusters >= len(data):
        st.warning("Number of clusters must be less than number of data points.")
        return np.zeros(len(data)), 0, 0
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels, kmeans.inertia_, silhouette_score(data, labels)

def run_pca(data, n_components):
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)
    return transformed, pca.explained_variance_ratio_

def run_agglomerative(data, n_clusters, linkage_method):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(data)
    return labels

def plot_elbow(data):
    inertias = []
    ks = range(2, min(11, len(data)))
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        inertias.append(kmeans.inertia_)
    fig, ax = plt.subplots()
    ax.plot(ks, inertias, '-o')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method')
    return fig

def plot_dendrogram(data):
    linked = linkage(data, method='ward')
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linked, ax=ax)
    ax.set_title('Hierarchical Clustering Dendrogram')
    return fig

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="clustered_data.csv">📥 Download Clustered Data</a>'
    return href

# -------------------
# Streamlit App
# -------------------
st.set_page_config(page_title="ClusterPlay", layout="wide")
st.title("🔍 ClusterPlay: Discover Patterns in Your Data")

st.sidebar.header("📁 Upload or Choose a Dataset")
sample_choice = st.sidebar.selectbox("Choose a sample dataset", ["None", "Iris", "Wine Quality"])
uploaded_file = st.sidebar.file_uploader("Or upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif sample_choice != "None":
    df = load_sample_data(sample_choice)
else:
    st.info("Please upload a dataset or select a sample to begin.")
    st.stop()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found for clustering.")
    st.stop()

st.sidebar.header("⚙️ Choose Algorithm")
model_type = st.sidebar.selectbox("Unsupervised Model", ["K-Means Clustering", "Hierarchical Clustering", "Principal Component Analysis (PCA)"])
selected_cols = st.multiselect("Select features for analysis:", numeric_cols, default=numeric_cols, help="Choose which columns to use in clustering or PCA.")

data = df[selected_cols].dropna()
df = df.copy()

st.header("📊 Data Preview")
st.write(df.head())

st.markdown("""
### 🧠 Learn More
**K-Means Clustering** partitions data into groups by minimizing intra-cluster variance. Useful in STEM for customer segmentation; in social science for grouping respondents.

**Hierarchical Clustering** builds a tree of clusters bottom-up. Used in genetics and social stratification studies.

**PCA** reduces dimensionality while retaining variance. Helpful for visualizing complex relationships.
""")

# -------------------
# K-Means Clustering Section
# -------------------
if model_type == "K-Means Clustering":
    k = st.sidebar.slider("Number of clusters (k)", 2, min(10, len(data)-1), 3, help="Adjust how many clusters to form.")
    labels, inertia, silhouette = run_kmeans(data, k)
    clustered_df = data.copy()
    clustered_df['Cluster'] = labels

    st.subheader("Clustered Data")
    fig = px.scatter_matrix(clustered_df, dimensions=selected_cols, color='Cluster', title="Cluster Visualization")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Silhouette Score:** {silhouette:.3f}")
    st.markdown(f"**Inertia (within-cluster sum of squares):** {inertia:.2f}")

    with st.expander("See Elbow Plot"):
        fig_elbow = plot_elbow(data)
        st.pyplot(fig_elbow)

    if len(selected_cols) >= 3:
        pca_data, _ = run_pca(data, 3)
        pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2', 'PC3'])
        pca_df['Cluster'] = labels
        st.subheader("🌍 3D PCA Cluster Visualization")
        fig3d = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='Cluster')
        st.plotly_chart(fig3d, use_container_width=True)

    with st.expander("Cluster Correlation Heatmap"):
        st.markdown("This shows how the average feature values compare across clusters.")
        heat_df = clustered_df.groupby('Cluster')[selected_cols].mean()
        fig, ax = plt.subplots()
        sns.heatmap(heat_df, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    df = clustered_df.copy()

# -------------------
# Hierarchical Clustering Section
# -------------------
elif model_type == "Hierarchical Clustering":
    k = st.sidebar.slider("Number of clusters (k)", 2, min(10, len(data)-1), 3, help="Choose the number of groups to extract from the hierarchy.")
    linkage_method = st.sidebar.selectbox("Linkage method", ["ward", "complete", "average"], help="Linkage strategy used to merge clusters.")
    labels = run_agglomerative(data, k, linkage_method)
    clustered_df = data.copy()
    clustered_df['Cluster'] = labels

    st.subheader("Clustered Data")
    fig = px.scatter_matrix(clustered_df, dimensions=selected_cols, color='Cluster', title="Cluster Visualization")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("See Dendrogram"):
        fig_dendro = plot_dendrogram(data)
        st.pyplot(fig_dendro)

    with st.expander("Cluster Correlation Heatmap"):
        st.markdown("This shows how the average feature values compare across clusters.")
        heat_df = clustered_df.groupby('Cluster')[selected_cols].mean()
        fig, ax = plt.subplots()
        sns.heatmap(heat_df, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    df = clustered_df.copy()

# -------------------
# Principal Component Analysis Section
# -------------------
elif model_type == "Principal Component Analysis (PCA)":
    n = st.sidebar.slider("Number of components", 2, min(5, len(selected_cols)), 2, help="Choose how many PCA dimensions to keep.")
    pca_data, variance = run_pca(data, n)
    pca_df = pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(n)])

    st.subheader("PCA Projection")
    fig = px.scatter(pca_df, x="PC1", y="PC2", title="PCA 2D Projection")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Explained Variance Ratio:**")
    st.bar_chart(variance)

# -------------------
# Download Clustered Dataset
# -------------------
st.markdown("""
## 📥 Download Results
Click below to export your clustered dataset:
""")
st.markdown(get_table_download_link(df), unsafe_allow_html=True)

if st.button("🎈 Celebrate Successful Clustering!"):
    st.balloons()

st.success("✅ Analysis complete. Adjust the model settings to explore different behaviors and visualize their effect!")
