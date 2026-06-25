# IrisVision-KMeans
An end-to-end Machine Learning project that performs unsupervised clustering on the Iris dataset using K-Means, feature scaling, and PCA for dimensionality reduction with interactive data visualization.
# 🌸 IrisVision-KMeans

An end-to-end Machine Learning project demonstrating **Unsupervised Learning** by clustering the famous **Iris Flower Dataset** using the **K-Means Clustering Algorithm**. The project also applies **StandardScaler** for feature normalization and **Principal Component Analysis (PCA)** for dimensionality reduction and visualization.

---

## 📌 Project Overview

This project groups iris flowers into three clusters based solely on their feature values without using the actual species labels during training. After clustering, PCA is applied to visualize the clusters in two dimensions and compare them with the original classes.

---

## 🎯 Objectives

* Learn the fundamentals of Unsupervised Machine Learning.
* Perform feature scaling using StandardScaler.
* Apply K-Means Clustering.
* Reduce dimensions using PCA.
* Visualize clusters.
* Compare predicted clusters with actual species.

---

## 📂 Dataset

**Dataset:** Iris Dataset

**Number of Samples:** 150

**Features**

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

**Target Classes**

* Setosa
* Versicolor
* Virginica

Dataset is loaded directly from **scikit-learn**, so no manual download is required.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Joblib

---

## 📈 Machine Learning Workflow

```text
Load Dataset
      │
      ▼
Explore Data
      │
      ▼
Feature Scaling
      │
      ▼
K-Means Clustering
      │
      ▼
Cluster Assignment
      │
      ▼
PCA (2D Projection)
      │
      ▼
Visualization
      │
      ▼
Compare with Actual Labels
      │
      ▼
Save Model
```

---

## 📊 Algorithms Used

### StandardScaler

Normalizes feature values so that each feature contributes equally during clustering.

### K-Means Clustering

Groups similar flowers into three clusters by minimizing the distance between samples and their assigned cluster centers.

### Principal Component Analysis (PCA)

Reduces four-dimensional feature space into two dimensions for easy visualization while preserving most of the information.

---

## 📁 Project Structure

```text
IrisVision-KMeans/
│
├── iris_clustering.ipynb
├── app.py
├── model.pkl
├── requirements.txt
├── README.md
├── images/
└── screenshots/
```

---

## ▶️ Installation

```bash
git clone https://github.com/your-username/IrisVision-KMeans.git

cd IrisVision-KMeans

pip install -r requirements.txt
```

---

## 🚀 Run the Project

```bash
python app.py
```

or

```bash
jupyter notebook
```

Open **iris_clustering.ipynb**.

---

## 📸 Output

* Dataset Exploration
* Cluster Centers
* PCA Visualization
* Predicted Clusters
* Actual Species Comparison

(Add screenshots here after completing the project.)

---

## 📚 Learning Outcomes

* Understanding Unsupervised Learning
* Feature Scaling
* Distance-based Clustering
* PCA for Visualization
* Machine Learning Workflow
* Data Visualization using Matplotlib

---

## 🔮 Future Improvements

* Interactive Streamlit Web App
* Elbow Method for Optimal K Selection
* Silhouette Score Evaluation
* Support for Custom CSV Upload
* Deploy using Streamlit Community Cloud

---

## 👨‍💻 Author

**Nitesh Barnwal**

B.Tech CSE | Machine Learning Enthusiast | GATE CS 2027 Aspirant

If you found this project helpful, consider giving it a ⭐ on GitHub.
