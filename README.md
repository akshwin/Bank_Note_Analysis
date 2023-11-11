# BANK NOTE ANALYSIS
### 1. Data Loading and Exploration:

In this machine learning project, the initial step involves loading the dataset and gaining an understanding of its structure. The data is loaded into a pandas DataFrame, a powerful data manipulation tool in Python. The dataset, obtained from the UCI Machine Learning Repository, comprises features such as variance, skewness, curtosis, entropy, and a class label indicating whether a bank note is genuine or fake.

The following Python code snippet demonstrates the data loading and initial exploration:

```python
import pandas as pd

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
column_names = ["variance", "skewness", "curtosis", "entropy", "class"]
df = pd.read_csv(url, names=column_names)

# Display basic information about the dataset
print(df.info())

# Display the first few rows of the dataset
print(df.head())
```

The `info()` method provides essential information such as the number of non-null entries and data types, offering a quick overview of the dataset. Meanwhile, the `head()` method displays the first few rows, giving an insight into the structure of the data.

### 2. Data Visualization:

Visualization is a crucial step in understanding the distribution of data and identifying potential patterns. In this project, various visualizations are employed to explore the dataset visually. The code uses libraries like matplotlib and seaborn for creating visual representations.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of the 'class' attribute
sns.countplot(x='class', data=df)
plt.title('Distribution of Genuine and Fake Bank Notes')
plt.show()

# Visualize histograms for individual features
df.hist(bins=20, figsize=(10, 8))
plt.suptitle('Histograms for Features')
plt.show()

# Visualize relationships between pairs of features using a pair plot
sns.pairplot(df, hue='class')
plt.suptitle('Pair Plot of Features')
plt.show()
```

The count plot provides insight into the balance of the classes, while histograms offer a glimpse into the distribution of individual features. The pair plot is particularly useful for identifying potential relationships and separations between features, especially concerning the target variable.

### 3. Data Preparation:

Before training machine learning models, it's essential to prepare the data. This involves splitting it into features (X) and the target variable (y). Additionally, standard scaling is applied to the features to ensure that they are on a comparable scale.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split the data into features (X) and target variable (y)
X = df.drop('class', axis=1)
y = df['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard scaling of features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The `train_test_split` function is employed to create training and testing sets, ensuring that the model's performance can be assessed on unseen data. Standard scaling is applied to prevent features with larger scales from dominating the learning process.

### 4. Model Implementation and Evaluation:

This project employs three different machine learning models: Logistic Regression, Support Vector Machine (SVM), and Random Forest Classifier. Each model is implemented, and its performance is evaluated using cross-validation.

#### Logistic Regression:

Logistic Regression is a fundamental classification algorithm that models the probability of the default class.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Logistic Regression
logreg = LogisticRegression(random_state=42)
logreg_scores = cross_val_score(logreg, X_train_scaled, y_train, cv=5, scoring='accuracy')
logreg_mean_accuracy = logreg_scores.mean()

print(f"Logistic Regression Mean Accuracy: {logreg_mean_accuracy:.2%}")
```

The code uses the `LogisticRegression` class from scikit-learn and employs cross-validation to obtain a more robust estimate of the model's performance.

#### Support Vector Machine (SVM):

SVM is a powerful algorithm for classification tasks. The code implements both a standard SVM and a hyperparameter-tuned SVM using grid search.

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Standard SVM
svm = SVC(random_state=42)
svm_scores = cross_val_score(svm, X_train_scaled, y_train, cv=5, scoring='accuracy')
svm_mean_accuracy = svm_scores.mean()

print(f"SVM Mean Accuracy: {svm_mean_accuracy:.2%}")

# Hyperparameter Tuned SVM using Grid Search
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001]}
grid_search_svm = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search_svm.fit(X_train_scaled, y_train)

best_svm = grid_search_svm.best_estimator_
best_svm_accuracy = grid_search_svm.best_score_

print(f"Best SVM Mean Accuracy after Hyperparameter Tuning: {best_svm_accuracy:.2%}")
```

The standard SVM is first implemented, and its mean accuracy is calculated. Subsequently, grid search is employed to find the best hyperparameters for the SVM model, enhancing its performance.

#### Random Forest Classifier:

Random Forest is an ensemble learning method that constructs multiple decision trees and merges them together.

```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_scores = cross_val_score(rf_classifier, X_train_scaled, y_train, cv=5, scoring='accuracy')
rf_mean_accuracy = rf_scores.mean()

print(f"Random Forest Mean Accuracy: {rf_mean_accuracy:.2%}")
```

The code utilizes the `RandomForestClassifier` class and cross-validation to assess the performance of the Random Forest model.

### 5. Evaluation Metrics:

Accuracy is a commonly used metric for classification tasks, providing the ratio of correctly predicted instances to the total number of instances. In addition to accuracy, confusion matrices and heatmaps are employed to visually represent the performance of the models on the test set.

```python
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

# Evaluate Logistic Regression on the test set
logreg.fit(X_train_scaled, y_train)
logreg_test_accuracy = logreg.score(X_test_scaled, y_test)

# Evaluate SVM on the test set
best_svm.fit(X_train_scaled, y_train)
svm_test_accuracy = best_svm.score(X_test_scaled, y_test)

# Evaluate Random Forest on the test set
rf_classifier.fit(X_train_scaled, y_train)
rf_test_accuracy = rf_classifier.score(X_test_scaled, y_test)

# Confusion matrices
logreg_confusion = confusion_matrix(y_test, logreg.predict(X_test_scaled))
svm_confusion = confusion_matrix(y_test, best_svm.predict(X_test_scaled))
rf

_confusion = confusion_matrix(y_test, rf_classifier.predict(X_test_scaled))

# Plot confusion matrices as heatmaps
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

plot_confusion_matrix(logreg, X_test_scaled, y_test, ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title('Logistic Regression')

plot_confusion_matrix(best_svm, X_test_scaled, y_test, ax=axes[1], cmap='Blues', values_format='d')
axes[1].set_title('SVM')

plot_confusion_matrix(rf_classifier, X_test_scaled, y_test, ax=axes[2], cmap='Blues', values_format='d')
axes[2].set_title('Random Forest')

plt.show()

print(f"Logistic Regression Test Accuracy: {logreg_test_accuracy:.2%}")
print(f"SVM Test Accuracy: {svm_test_accuracy:.2%}")
print(f"Random Forest Test Accuracy: {rf_test_accuracy:.2%}")
```

The confusion matrices provide a detailed breakdown of true positives, true negatives, false positives, and false negatives for each model. Visualizing these matrices as heatmaps offers a clear representation of the models' performance.

### 6. Summary of Results:

After implementing and evaluating the three models, the project summarizes the results, highlighting the mean accuracy achieved during cross-validation and the accuracy on the test set for each model.

```python
print("Summary of Results:")
print(f"Logistic Regression Mean Accuracy: {logreg_mean_accuracy:.2%}")
print(f"SVM Mean Accuracy: {svm_mean_accuracy:.2%}")
print(f"Best SVM Mean Accuracy after Hyperparameter Tuning: {best_svm_accuracy:.2%}")
print(f"Random Forest Mean Accuracy: {rf_mean_accuracy:.2%}")

print("\nTest Set Accuracy:")
print(f"Logistic Regression: {logreg_test_accuracy:.2%}")
print(f"SVM: {svm_test_accuracy:.2%}")
print(f"Random Forest: {rf_test_accuracy:.2%}")
```

This section provides a concise summary of the models' performance, allowing for a quick comparison of their accuracies.

### Conclusion:

In conclusion, this machine learning project involves the classification of bank notes as genuine or fake using various models. The code covers data loading, exploration, visualization, data preparation, model implementation, and evaluation. By employing different models and visualization techniques, the project aims to provide a comprehensive understanding of the dataset and the performance of each model.
