import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("student_dropout.csv")

print("Dataset Loaded Successfully")
print(df.head())

# -----------------------------
# Handle Missing Values
# -----------------------------
df.fillna(df.mode().iloc[0], inplace=True)

# -----------------------------
# Encode Categorical Columns
# -----------------------------
label_encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop("Dropout", axis=1)
y = df["Dropout"]

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save Scaler
joblib.dump(scaler, "scaler.pkl")

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Models
# -----------------------------
models = {

    "Logistic Regression": LogisticRegression(),

    "Decision Tree": DecisionTreeClassifier(),

    "Random Forest": RandomForestClassifier(),

    "SVM": SVC(probability=True),

    "KNN": KNeighborsClassifier(),

    "Naive Bayes": GaussianNB()

}

best_model = None
best_accuracy = 0

print("\nModel Comparison\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print(f"{name} : {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print("\nBest Accuracy :", best_accuracy)

# -----------------------------
# Save Best Model
# -----------------------------
joblib.dump(best_model, "best_model.pkl")

print("\nModel Saved Successfully!")

# -----------------------------
# Final Report
# -----------------------------
prediction = best_model.predict(X_test)

print("\nClassification Report\n")

print(classification_report(y_test, prediction))