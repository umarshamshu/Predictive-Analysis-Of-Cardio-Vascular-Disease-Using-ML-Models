

#  Predictive Analysis of Cardiovascular Disease Using Machine Learning Models

## 📌 Project Overview

Cardiovascular diseases (CVDs), particularly **Coronary Heart Disease (CHD)**, are among the leading causes of mortality worldwide. Early detection plays a crucial role in reducing fatal outcomes.

This project applies **machine learning techniques** to predict the risk of CHD using clinical and lifestyle-related features from the **Framingham Heart Study dataset**. Multiple classification models are implemented and evaluated to identify the most effective algorithm for accurate prediction.

---

## 🎯 Objectives

* To analyze cardiovascular risk factors using data-driven techniques
* To build and compare multiple machine learning classification models
* To handle class imbalance using **SMOTE**
* To identify the best-performing model based on accuracy and recall
* To support early diagnosis and clinical decision-making

---

#Here is the **updated GitHub-ready content**, with your requested changes applied **only to the “Data Description” section**.
You can directly **replace this section** in your existing `README.md`.

---

## 📂 Dataset Description (Data Source)

* **Dataset Name**: Heart Disease Prediction Dataset (Framingham Study)
* **Source Platform**: Kaggle
* **Dataset Link**:
  👉 [https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression](https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression)

### 📊 Dataset Overview

This dataset is derived from the **Framingham Heart Study** and is widely used for predicting **Coronary Heart Disease (CHD)** using statistical and machine learning models.

* **Number of Observations**: **4,238**
* **Number of Variables (Features)**: **16**
* **Target Variable**: **CHD**

  * `0` → No Coronary Heart Disease
  * `1` → Presence of Coronary Heart Disease

The dataset includes a combination of **demographic, behavioral, and medical risk factors**, making it suitable for cardiovascular disease prediction and risk assessment.

---

### 🧾 Variable Description

| Variable Name      | Description                                         | Type       |
| ------------------ | --------------------------------------------------- | ---------- |
| `sex`              | Gender of the individual (0 = Female, 1 = Male)     | Binary     |
| `age`              | Age of the individual (in years)                    | Continuous |
| `currentSmoker`    | Whether the individual is a current smoker          | Binary     |
| `cigsPerDay`       | Number of cigarettes smoked per day                 | Continuous |
| `BPMeds`           | Whether the patient is on blood pressure medication | Binary     |
| `prevalentStroke`  | History of stroke                                   | Binary     |
| `prevalentHyp`     | History of hypertension                             | Binary     |
| `diabetes`         | Diabetes status                                     | Binary     |
| `totChol`          | Total cholesterol level                             | Continuous |
| `sysBP`            | Systolic blood pressure                             | Continuous |
| `diaBP`            | Diastolic blood pressure                            | Continuous |
| `BMI`              | Body Mass Index                                     | Continuous |
| `heartRate`        | Heart rate                                          | Continuous |
| `glucose`          | Glucose level                                       | Continuous |
| `TenYearCHD (CHD)` | 10-year risk of Coronary Heart Disease              | Binary     |

---

### 📌 Why This Dataset?

* Contains **real-world clinical data**
* Suitable for **binary classification**
* Includes **imbalanced target classes**, enabling the use of techniques like **SMOTE**
* Widely accepted for **academic and research projects**

---

### Key Features

| Feature          | Description                  |
| ---------------- | ---------------------------- |
| Age              | Age of the individual        |
| Sex              | 0 = Female, 1 = Male         |
| Current Smoker   | Smoking status               |
| Cigs Per Day     | Cigarettes smoked per day    |
| BPMeds           | On blood pressure medication |
| Prevalent Stroke | History of stroke            |
| Prevalent Hyp    | History of hypertension      |
| Diabetes         | Diabetes status              |
| Tot Chol         | Total cholesterol            |
| Sys BP           | Systolic blood pressure      |
| Dia BP           | Diastolic blood pressure     |
| BMI              | Body Mass Index              |
| Heart Rate       | Heart rate                   |
| Glucose          | Glucose level                |

---

## 🧹 Data Preprocessing

* Handling missing values

  * Mean imputation for numerical variables
  * Mode imputation for categorical variables
* Feature scaling and normalization
* Outlier handling
* **Class imbalance correction using SMOTE**

---

## 📊 Exploratory Data Analysis (EDA)

Key insights from EDA:

* Males show higher CHD prevalence than females
* Diabetic individuals have significantly higher CHD risk
* Patients with hypertension and stroke history are at greater risk
* BP medication users show increased CHD prevalence

EDA visualizations include:

* Bar plots
* Distribution plots
* Correlation heatmaps

---

## Machine Learning Models Used

* **Logistic Regression**
* **Decision Tree**
* **Random Forest**
* **K-Nearest Neighbors (KNN)**
* **Support Vector Machine (SVM)**
* **XGBoost**

Each model was trained on:

* **Imbalanced dataset**
* **Balanced dataset (SMOTE applied)**

---

## Model Performance Summary

| Model               | Best Accuracy       | Remarks                  |
| ------------------- | ------------------- | ------------------------ |
| Decision Tree       | 82.79% (Imbalanced) | Poor minority detection  |
| Logistic Regression | 86.16% (Imbalanced) | Low recall for CHD       |
| KNN                 | 84% (Balanced)      | Better class balance     |
| SVM                 | 84% (Balanced)      | Robust classification    |
| Random Forest       | 86% (Imbalanced)    | Strong generalization    |
| **XGBoost**         | **89% (Balanced)**  | Best overall performance |

---

## Final Verdict

**Random Forest and XGBoost** emerged as the most effective models for predicting Coronary Heart Disease.

* XGBoost achieved the **highest accuracy (89%)** with strong minority-class detection
* Random Forest demonstrated reliable generalization
* Balanced datasets improved recall and fairness

This study highlights the **transformative role of machine learning in healthcare**, especially for early risk assessment and clinical support systems.

---

## Tools & Technologies

* **Programming Language**: Python
* **Libraries**:

  * NumPy
  * Pandas
  * Matplotlib
  * Seaborn
  * Scikit-learn
  * Imbalanced-learn (SMOTE)
  * XGBoost
