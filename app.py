import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# ----------------------------
# Load Files
# ----------------------------
df = pd.read_csv("student_dropout.csv")
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🎓 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Dataset",
        "Visualization",
        "Prediction",
        "About"
    ]
)

# ----------------------------
# HOME
# ----------------------------
if page == "Home":

    st.title("🎓 Student Dropout Prediction System")

    st.markdown("---")

    st.subheader("Project Overview")

    st.write("""
This project predicts whether a student is likely to:

✅ Continue Education

or

❌ Drop Out

using Machine Learning Classification Algorithms.
""")

    c1, c2, c3 = st.columns(3)

    c1.metric("Dataset Rows", df.shape[0])
    c2.metric("Features", df.shape[1]-1)
    c3.metric("Target", "Dropout")

# ----------------------------
# DATASET
# ----------------------------
elif page == "Dataset":

    st.title("📊 Dataset Explorer")

    st.write(df.head())

    st.subheader("Shape")

    st.write(df.shape)

    st.subheader("Columns")

    st.write(df.columns)

    st.subheader("Statistics")

    st.write(df.describe())

# ----------------------------
# VISUALIZATION
# ----------------------------
elif page == "Visualization":

    st.title("📈 Data Visualization")

    fig = px.histogram(
        df,
        x="CGPA",
        color="Dropout",
        title="CGPA Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(
        df,
        x="Attendance",
        color="Dropout",
        title="Attendance Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        df,
        x="Study_Hours",
        y="CGPA",
        color="Dropout"
    )

    st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# PREDICTION
# ----------------------------
elif page == "Prediction":

    st.title("🤖 Student Prediction")

    age = st.slider("Age",17,30,20)

    gender = st.selectbox("Gender",["Male","Female"])

    attendance = st.slider("Attendance",40,100,80)

    study = st.slider("Study Hours",1,8,4)

    previous = st.slider("Previous Grade",40,100,70)

    assignment = st.slider("Assignments Submitted",0,10,5)

    income = st.number_input("Family Income",10000,100000,50000)

    internet = st.selectbox("Internet",["Yes","No"])

    parent = st.selectbox(
        "Parental Education",
        [
            "School",
            "Graduate",
            "Postgraduate"
        ]
    )

    distance = st.slider("Distance",1,30,10)

    scholarship = st.selectbox("Scholarship",["Yes","No"])

    extra = st.selectbox("Extracurricular",["Yes","No"])

    semester = st.slider("Semester",1,8,4)

    cgpa = st.slider("CGPA",4.0,10.0,7.0)

    backlog = st.slider("Backlogs",0,5,0)

    if st.button("Predict"):

        gender = 1 if gender=="Male" else 0
        internet = 1 if internet=="Yes" else 0
        scholarship = 1 if scholarship=="Yes" else 0
        extra = 1 if extra=="Yes" else 0

        if parent=="School":
            parent=0
        elif parent=="Graduate":
            parent=1
        else:
            parent=2

        sample = [[
            age,
            gender,
            attendance,
            study,
            previous,
            assignment,
            income,
            internet,
            parent,
            distance,
            scholarship,
            extra,
            semester,
            cgpa,
            backlog
        ]]

        sample = scaler.transform(sample)

        prediction = model.predict(sample)[0]

        probability = model.predict_proba(sample)[0]

        if prediction==1:

            st.error("⚠ Student is likely to Drop Out")

        else:

            st.success("✅ Student is likely to Continue")

        st.subheader("Probability")

        st.write(probability)

# ----------------------------
# ABOUT
# ----------------------------
else:

    st.title("📚 About Classification")

    st.write("""
Classification is a Machine Learning technique used to predict categories.

Algorithms Used:

• Logistic Regression

• Decision Tree

• Random Forest

• SVM

• KNN

• Naive Bayes
""")