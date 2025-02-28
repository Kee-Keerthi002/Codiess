#!/usr/bin/env python
# coding: utf-8

# In[3]:


#SVM for Linear dataset


# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load a linear dataset
data = datasets.load_iris()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear SVM model
linear_svm = SVC(kernel='linear')

# Train the model
linear_svm.fit(X_train, y_train)

# Make predictions on the test set
predictions = linear_svm.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Linear SVM Accuracy: {accuracy}")


# In[6]:


#SVM for Non-Linear dataset


# Import necessary libraries
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
# Create a non-linear dataset
X, y = make_circles(n_samples=100, factor=0.5, noise=0.1)

# Plot the non-linear dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.title("Non-linear Dataset")
plt.show()

# Create a non-linear SVM model
non_linear_svm = SVC(kernel='rbf')

# Train the model
non_linear_svm.fit(X, y)

# Visualize the decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = non_linear_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])

# Plot decision boundary and margins
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
            linestyles=['--', '-', '--'])
plt.title("Non-linear SVM Decision Boundary")
plt.show()

