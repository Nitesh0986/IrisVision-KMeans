
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="IrisVision-KMeans",
    page_icon="🌸",
    layout="wide"
)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["Species"] = iris.target

species_names = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

df["Species Name"] = df["Species"].map(species_names)

# ---------------------------------------------------
# Data Preprocessing
# ---------------------------------------------------
X = df.iloc[:, :-2]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.title("🌸 IrisVision-KMeans")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset",
        "📈 EDA",
        "🤖 K-Means",
        "📉 PCA",
        "ℹ️ About"
    ]
)

# ---------------------------------------------------
# HOME
# ---------------------------------------------------
if page == "🏠 Home":

    st.title("🌸 IrisVision-KMeans")

    st.write("""
    ## Welcome!

    This project demonstrates **Unsupervised Machine Learning**
    using the famous **Iris Dataset**.

    ### Algorithms Used

    - StandardScaler
    - K-Means Clustering
    - Principal Component Analysis (PCA)

    ### Tech Stack

    - Python
    - Pandas
    - Scikit-Learn
    - Matplotlib
    - Streamlit
    """)

# ---------------------------------------------------
# DATASET
# ---------------------------------------------------
elif page == "📊 Dataset":

    st.title("📊 Iris Dataset")

    st.dataframe(df)

    st.write("Dataset Shape:", df.shape)

# ---------------------------------------------------
# EDA
# ---------------------------------------------------
elif page == "📈 EDA":

    st.title("📈 Exploratory Data Analysis")

    st.subheader("Dataset Statistics")

    st.write(df.describe())

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(8,6))

    sns.heatmap(
        df.iloc[:,0:4].corr(),
        annot=True,
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)

# ---------------------------------------------------
# K-MEANS
# ---------------------------------------------------
elif page == "🤖 K-Means":

    st.title("🤖 K-Means Clustering")

    st.subheader("Cluster Distribution")

    st.bar_chart(df["Cluster"].value_counts())

    st.subheader("Cluster Centers")

    st.write(kmeans.cluster_centers_)

# ---------------------------------------------------
# PCA
# ---------------------------------------------------
elif page == "📉 PCA":

    st.title("📉 PCA Visualization")

    fig, ax = plt.subplots(figsize=(8,6))

    scatter = ax.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=clusters,
        cmap="viridis",
        s=80
    )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("K-Means Clusters using PCA")

    plt.colorbar(scatter)

    st.pyplot(fig)


# ---------------------------------------------------
# ABOUT
# ---------------------------------------------------
elif page == "ℹ️ About":

    st.title("ℹ️ About Project")

    st.write("""
## 🌸 Project Objective

Build an end-to-end Machine Learning project using the Iris Dataset.

---

## 📌 Workflow

- Load Iris Dataset
- Exploratory Data Analysis (EDA)
- Feature Selection
- Feature Scaling using StandardScaler
- K-Means Clustering
- PCA Visualization
- Model Saving using Joblib
- Streamlit Dashboard

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Streamlit

---

## 👨‍💻 Developed By

**Nitesh Barnwal**

B.Tech Computer Science & Engineering

Narula Institute of Technology

🎯 Preparing for GATE CS 2027

💡 Aspiring Machine Learning Engineer

⭐ GitHub Project: IrisVision-KMeans
""")

