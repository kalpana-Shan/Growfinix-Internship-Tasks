# pip install pandas scikit-learn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset (download heart.csv from Kaggle "Heart Disease UCI" and place it here)
df = pd.read_csv("heart.csv")

# Basic cleaning
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)  # encode categoricals if any

print(df.columns.tolist())
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling (the core learning goal of this task)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# Model 2: SVM
svm = SVC(kernel="rbf")
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)

print("\n=== SVM (RBF) ===")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred))