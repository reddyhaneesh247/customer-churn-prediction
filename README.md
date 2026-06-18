# Customer Churn Prediction

## Overview

This project predicts whether a customer is likely to churn (leave the service) using machine learning techniques. The workflow includes data preprocessing, handling class imbalance with SMOTE, model training, evaluation, and model persistence for deployment.

## Features

* Data cleaning and preprocessing
* Handling categorical and numerical features using preprocessing pipelines
* Class imbalance treatment using SMOTE
* Training and evaluation of multiple machine learning models
* Hyperparameter tuning for improved performance
* Model serialization using Joblib
* Customer churn prediction on new data

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* XGBoost
* Joblib
* Matplotlib
* Seaborn

## Dataset

The dataset contains customer demographic, account, and banking information such as:

* Credit Score
* Geography/Nationality
* Gender
* Age
* Tenure
* Account Balance
* Number of Products
* Credit Card Ownership
* Active Membership Status
* Estimated Salary

Target Variable:

* Exited (0 = Customer Stays, 1 = Customer Churns)

## Project Workflow

### 1. Data Preprocessing

* Handled missing values and data inconsistencies
* Encoded categorical features
* Scaled numerical features
* Built preprocessing pipelines using ColumnTransformer

### 2. Class Imbalance Handling

The dataset exhibited class imbalance. SMOTE (Synthetic Minority Over-sampling Technique) was applied to generate synthetic minority-class samples and improve model learning.

### 3. Model Training

The following models were trained and evaluated:

* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier
* XGBoost Classifier

### 4. Best Model

The best-performing model was a Gradient Boosting Classifier integrated within a machine learning pipeline.

### 5. Model Persistence

The trained machine learning pipeline was serialized using Joblib for efficient storage and deployment.

```python
from joblib import dump

dump(gb_best, "customer_churn_pipeline.pkl")
```

## Prediction Pipeline

The saved pipeline automatically:

1. Applies preprocessing transformations
2. Uses the trained model for prediction
3. Returns churn predictions for new customer records

## Results

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

Performance comparison was conducted to select the most effective model for churn prediction.

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/reddyhaneesh247/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Open the Notebook in Google Colab

Upload or open:

```text
Customer_churn.ipynb
```

### 3. Install Required Libraries

```python
!pip install pandas numpy scikit-learn imbalanced-learn xgboost joblib
```

### 4. Upload the Dataset

Upload the dataset file to Google Colab and update the dataset path if necessary.

### 5. Run All Cells

Execute all notebook cells sequentially to:

* Perform data preprocessing
* Handle class imbalance using SMOTE
* Train and evaluate multiple machine learning models
* Select the best-performing model
* Generate churn predictions

### 6. Save the Trained Pipeline

```python
from joblib import dump

dump(gb_best, "customer_churn_pipeline.pkl")
```

### 7. Load the Pipeline for Inference

```python
from joblib import load

model = load("customer_churn_pipeline.pkl")
```

### Output

The trained model predicts customer churn:

* **0 → Customer Stays**
* **1 → Customer Churns**

The model predicts:

* **0 → Customer Stays**
* **1 → Customer Churns**

### THE END

