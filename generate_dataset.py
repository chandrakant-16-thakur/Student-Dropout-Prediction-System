import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

data = {
    "Age": np.random.randint(17, 26, n),
    "Gender": np.random.choice(["Male", "Female"], n),
    "Attendance": np.random.randint(40, 100, n),
    "Study_Hours": np.random.randint(1, 8, n),
    "Previous_Grade": np.random.randint(40, 100, n),
    "Assignments_Submitted": np.random.randint(0, 10, n),
    "Family_Income": np.random.randint(10000, 100000, n),
    "Internet_Access": np.random.choice(["Yes", "No"], n),
    "Parental_Education": np.random.choice(
        ["School", "Graduate", "Postgraduate"], n
    ),
    "Distance_From_College": np.random.randint(1, 30, n),
    "Scholarship": np.random.choice(["Yes", "No"], n),
    "Extracurricular": np.random.choice(["Yes", "No"], n),
    "Semester": np.random.randint(1, 9, n),
    "CGPA": np.round(np.random.uniform(4.0, 10.0, n), 2),
    "Backlogs": np.random.randint(0, 6, n),
}

# Generate target column
dropout = []

for i in range(n):
    risk = 0

    if data["Attendance"][i] < 60:
        risk += 1
    if data["Study_Hours"][i] < 3:
        risk += 1
    if data["CGPA"][i] < 6:
        risk += 1
    if data["Backlogs"][i] >= 3:
        risk += 1

    dropout.append(1 if risk >= 2 else 0)

data["Dropout"] = dropout

df = pd.DataFrame(data)

df.to_csv("student_dropout.csv", index=False)

print("✅ Dataset generated successfully!")
print(df.head())