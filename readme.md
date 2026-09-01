# ❤️ Heart Disease Prediction

A machine learning web application that predicts the likelihood of heart disease based on a patient's clinical and physiological information.

The project uses **K-Nearest Neighbors (KNN)** as the prediction model and provides an interactive interface built with **Streamlit**.

---

## 🚀 Live Project

> Run the application locally using Streamlit.

```bash
streamlit run app.py
```

---

## 📌 Project Overview

Heart disease is one of the major health concerns worldwide. Machine learning can be used to analyze patient information and identify patterns associated with heart disease.

This project takes several patient-related features as input and uses a trained machine learning model to classify the patient into:

* 🟢 **Low Risk** — Predicted absence of heart disease
* 🔴 **High Risk** — Predicted presence of heart disease

The application provides a simple and user-friendly interface where users can enter patient information and instantly receive a model prediction.

> ⚠️ This project is intended for educational and demonstration purposes only and should not be used as a medical diagnosis system.

---

# ✨ Features

* ❤️ Heart disease risk prediction
* 🤖 Machine learning-based classification
* 📊 KNN classification model
* 📏 Feature scaling using StandardScaler
* 🔢 Categorical feature encoding
* 🖥️ Interactive Streamlit web interface
* 📋 User-friendly input form
* 📈 Prediction probability
* 🔎 View entered patient information
* ⚡ Fast local prediction
* 💾 Saved model and preprocessing objects using Joblib

---

# 🧠 Machine Learning Workflow

The project follows a standard machine learning pipeline:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Handle Missing Values
     ↓
Categorical Encoding
     ↓
Train-Test Split
     ↓
Feature Scaling
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Save Model
     ↓
Streamlit Application
     ↓
User Input
     ↓
Preprocessing
     ↓
KNN Prediction
     ↓
Risk Result
```

---

# 📊 Dataset

The dataset contains clinical information about patients and a target variable indicating the presence of heart disease.

### Features

| Feature          | Description                           |
| ---------------- | ------------------------------------- |
| `Age`            | Age of the patient                    |
| `Sex`            | Gender of the patient                 |
| `ChestPainType`  | Type of chest pain                    |
| `RestingBP`      | Resting blood pressure                |
| `Cholesterol`    | Serum cholesterol level               |
| `FastingBS`      | Fasting blood sugar indicator         |
| `RestingECG`     | Resting electrocardiogram result      |
| `MaxHR`          | Maximum heart rate achieved           |
| `ExerciseAngina` | Exercise-induced angina               |
| `Oldpeak`        | ST depression caused by exercise      |
| `ST_Slope`       | Slope of the peak exercise ST segment |
| `HeartDisease`   | Target variable                       |

### Target

```text
0 → No heart disease
1 → Heart disease
```

---

# 🔍 Exploratory Data Analysis

Several techniques were used during EDA to understand relationships between features and the target.

Examples include:

### Correlation Analysis

```python
df.corr(numeric_only=True)
```

Correlation helps identify relationships between numerical features.

### Numerical Feature Visualization

Scatter plots can be used to examine relationships such as:

* Age vs Heart Disease
* Cholesterol vs Heart Disease
* MaxHR vs Heart Disease
* Oldpeak vs Heart Disease

### Categorical Feature Analysis

Box plots can be used to compare distributions across categories:

```python
sns.boxplot(
    data=df,
    x="ChestPainType",
    y="Age"
)
```

---

# 🤖 Models Evaluated

Multiple classification algorithms were evaluated during the project:

| Model               | Purpose                       |
| ------------------- | ----------------------------- |
| Logistic Regression | Linear classification         |
| KNN                 | Distance-based classification |
| Naive Bayes         | Probabilistic classification  |
| Decision Tree       | Rule-based classification     |
| SVM                 | Margin-based classification   |

The final application uses **KNN** as the deployed model.

---

# 🏆 Model Evaluation

The models were evaluated using:

### Accuracy

Measures the proportion of correct predictions.

```text
Accuracy = Correct Predictions / Total Predictions
```

### F1 Score

F1 Score combines:

* Precision
* Recall

It is particularly useful when both false positives and false negatives matter.

Example evaluation structure:

```python
accuracy_score(y_test, y_pred)

f1_score(y_test, y_pred)
```

---

# 📏 Feature Scaling

Because KNN is a distance-based algorithm, feature scaling is especially important.

The project uses a scaler before making predictions:

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The trained scaler is saved and reused in the Streamlit application.

---

# 🔢 Categorical Encoding

Categorical features such as:

```text
Sex
ChestPainType
RestingECG
ExerciseAngina
ST_Slope
```

are converted into numerical representations before being passed to the machine learning model.

The application recreates the same feature structure expected by the trained model.

---

# 💾 Saved Machine Learning Files

The project uses Joblib to save the trained model and preprocessing objects.

```text
KNN_heart.pkl
scaler.pkl
columns.pkl
```

### `KNN_heart.pkl`

Contains the trained KNN classifier.

### `scaler.pkl`

Contains the fitted feature scaler used during training.

### `columns.pkl`

Contains the expected feature/column order required by the model.

Keeping the same preprocessing pipeline during training and prediction helps prevent feature mismatch.

---

# 🖥️ Streamlit Application

The web application provides input controls for all required patient features.

Users can enter:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise Angina
* Oldpeak
* ST Slope

After clicking:

```text
🔍 Predict Heart Disease
```

the application processes the input and displays the prediction.

### Example

```text
Patient Input
     ↓
Create DataFrame
     ↓
Match Expected Columns
     ↓
Scale Features
     ↓
KNN Model
     ↓
Prediction
     ↓
Low Risk / High Risk
```

---

# 📁 Project Structure

```text
Heart_disease_prediction/
│
├── app.py
│
├── KNN_heart.pkl
├── scaler.pkl
├── columns.pkl
│
├── heart.csv
│
└── README.md
```

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* K-Nearest Neighbors
* Logistic Regression
* Naive Bayes
* Decision Tree
* Support Vector Machine

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Deployment / UI

* Streamlit

### Model Serialization

* Joblib

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd Heart_disease_prediction
```

---

## 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

Usually, Streamlit runs locally at:

```text
http://localhost:8501
```

---

# 🧪 Example Inputs

### 🔴 High-Risk Example

```text
Age: 65
Sex: M
Chest Pain Type: ASY
Resting BP: 160
Cholesterol: 300
Fasting BS: 1
Resting ECG: ST
Max HR: 110
Exercise Angina: Y
Oldpeak: 2.5
ST Slope: Flat
```

### 🟢 Low-Risk Example

```text
Age: 30
Sex: F
Chest Pain Type: ATA
Resting BP: 110
Cholesterol: 170
Fasting BS: 0
Resting ECG: Normal
Max HR: 180
Exercise Angina: N
Oldpeak: 0.0
ST Slope: Up
```

> The prediction depends on the trained model and should not be interpreted as a medical diagnosis.

---

# 📈 Future Improvements

Possible improvements for future versions include:

* Deploy the application online
* Add model comparison inside the application
* Add confusion matrix visualization
* Add ROC-AUC evaluation
* Add feature importance/explainability
* Improve UI with interactive charts
* Add prediction history
* Add downloadable prediction reports
* Use a machine learning pipeline to combine preprocessing and prediction
* Add automated model retraining

---

# 🔐 Disclaimer

This project is created for **educational and machine learning demonstration purposes**.

The prediction generated by this application is based on patterns learned from the training dataset. It is **not a substitute for professional medical advice, diagnosis, or treatment**.

---

# 👨‍💻 Author

**Vedant Contractor**

Computer Engineering Student

---

## ⭐ If you found this project useful

Give the repository a ⭐ on GitHub and feel free to explore, improve, and contribute to the project.
