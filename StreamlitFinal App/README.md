# ClusterPlay: Discover Patterns in Your Data

## 📌 Project Overview

**ClusterPlay** is an interactive Streamlit web app that enables users to explore unsupervised machine learning methods on any tabular dataset. With no coding required, users can upload datasets or use built-in examples, apply clustering and dimensionality reduction techniques, and interpret results through intuitive visualizations.

This app is ideal for data science learners, educators, and anyone curious about exploring hidden patterns in data using:

* **K-Means Clustering**
* **Hierarchical Clustering**
* **Principal Component Analysis (PCA)**

---

## 🌐 Live App

🎯 **Try It Out Here**: [ClusterPlay Streamlit App](https://jthomp33-thompson-data-science-streamlitfinalappfinalapp-davhqo.streamlit.app/)

> Upload your own CSV file or explore sample datasets like Iris or Wine Quality.
> Adjust parameters live, and instantly visualize clustering behaviors, performance metrics, and dimensional reductions in real time.

✅ Meets all expectations for interactivity, transparency, and usability.
🚀 Deployed on Streamlit Community Cloud — no installation required.

---

## 🚀 Instructions

### Run Locally

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/THOMPSON-Data-Science-Portfolio.git
   cd THOMPSON-Data-Science-Portfolio/StreamlitFinalApp
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the app**:

   ```bash
   streamlit run finalapp.py
   ```

---

## 🧠 App Features

### 📁 Dataset Handling

* Upload your own CSV or choose from sample datasets.
* Numeric columns are auto-selected for analysis.
* Preview the first few rows in the interface.

---

## 🔬 Models Implemented and Explained

### 1. **K-Means Clustering**

Segment your dataset into *k* clusters using distance-based partitioning.

**Features:**

* Choose number of clusters (`k`)
* View interactive scatter matrix labeled by cluster
* Analyze clustering performance:

  * **Silhouette Score** – how distinct and well-separated clusters are
  * **Inertia** – internal cluster cohesion
* **Elbow Plot** – visual method to find optimal `k`
* **3D PCA Projection** – spatial view of cluster structure
* **Cluster Heatmap** – compare average feature values across clusters

🛠️ *Tunable Parameter:*

* `k`: Number of clusters

---

### 2. **Hierarchical Clustering**

Builds a hierarchy of clusters using bottom-up agglomeration.

**Features:**

* Choose number of clusters and linkage method (`ward`, `average`, `complete`)
* Interactive scatter matrix with cluster labels
* **Dendrogram** – visualize how clusters merge
* **Cluster Heatmap** – understand feature differences by cluster group

🛠️ *Tunable Parameters:*

* `k`: Number of clusters
* `linkage method`: Strategy to measure distances between clusters

---

### 3. **Principal Component Analysis (PCA)**

Reduces high-dimensional data while preserving variance for visualization.

**Features:**

* Choose number of components (`n`)
* View 2D projection using the first two principal components
* **Explained Variance Chart** – shows how much of the dataset’s variance is captured by each component

🛠️ *Tunable Parameter:*

* `n_components`: Number of PCA axes to retain

---

## 📊 Visualization Highlights

| Tool                 | What It Shows                      | Why It Matters                                  |
| -------------------- | ---------------------------------- | ----------------------------------------------- |
| **Silhouette Score** | Compactness/separation of clusters | Guides whether your clusters are meaningful     |
| **Inertia**          | Internal cohesion                  | Helps detect overfitting in K-Means             |
| **Elbow Plot**       | Optimal cluster number (`k`)       | Visual aid for choosing `k`                     |
| **Dendrogram**       | Cluster merging process            | Informs cut-off point in hierarchy              |
| **3D PCA Plot**      | Spatial view of clusters           | Great for spotting overlap and outliers         |
| **Cluster Heatmap**  | Feature averages by cluster        | Like a confusion matrix for unsupervised models |
| **Variance Chart**   | PCA compression quality            | Tells you how much signal you're keeping        |

---

## 📚 References

* [Scikit-learn documentation](https://scikit-learn.org/stable/)
* [Streamlit documentation](https://docs.streamlit.io/)
* [Plotly Express](https://plotly.com/python/plotly-express/)
* [Seaborn Heatmaps](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
* [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
* Iris dataset via `sklearn.datasets`

---

## 🧼 Code Quality & Contributions

* Modular helper functions and well-commented code
* Clean, logical structure: helpers → algorithms → UI
* Designed for extensibility (e.g., adding DBSCAN or t-SNE)

---

## 📁 Project Structure

```
├── finalapp.py              # Main Streamlit app
├── requirements.txt         # Dependencies
├── README.md                # This file
```

---

## 🙌 Acknowledgments

Created as a final project for Notre Dame’s **Introduction to Data Science** course.
**Author:** James W. Thompson II

---
