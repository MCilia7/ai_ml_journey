# This file contains the iris flower dataset, which contains 150 records of the three types of iris flowers - Setosa, Versicolor, and Virginica. For each flower, you are given four measurements: sepal length, sepal width, petal length, and petal width.
# This is a Supervised Learning Classification problem 

# ML Workflow:
# Data Collection -> Data Preprocessing -> Train/Test Split -> Model Training -> Model Evaluation -> Inference
# ==============================================================================================================

# ------------------------------------------------------------------
# TASK: Predict the flower type given four features in a dataset    |
# ------------------------------------------------------------------

# Step 1: Load the data
from sklearn.datasets import load_iris 
iris = load_iris()

# Step 2: Data Preprocessing
# Convert to Pandas DataFrame
import pandas as pd
X = pd.DataFrame(iris.data, columns=iris.feature_names) # these are the features
y = iris.target # this contains the labels


# Understand the data before modelling using class distribution
import seaborn as sns
sns.countplot(x=iris.target)

# Use pair plots to see relationships between features
sns.pairplot(df, hue="target")

# Use a correlation heatmap to quantify any patterns
sns.heatmap(df.corr(),annot=True)

# Step 3: Train/Test split
# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scaling will transform your features to have mean=0 and standard deviation=1

# Before Scaling
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(14,5))
ax[0].hist(X.iloc[:,0], bins=20, color="skyblue")
ax[0].set_title("Before Scaling: Sepal Length")

ax[0].hist(X_train[:,0], bins=20, color="orange")
ax[0].set_title("After Scaling: Sepal Length")
plt.show()

# Step 4: Model Training
# K-Nearest Neighbours (KNN)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Decision Tree (splits the dataset recursively based on feature thresholds)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(X_train_scaled, y_train)

# Step 5: Evaluation
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, knn.predict(X_test_scaled)))
print(classification_report(y_test, knn.predict(X_test_scaled)))
