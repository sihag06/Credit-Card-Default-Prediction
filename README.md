# 💳 Credit Card Default Prediction

## 📜 Project Overview

This project aims to develop a robust machine learning model to **predict whether a credit card customer will default on their payment in the next month**.  
By accurately identifying high-risk customers, financial institutions can take proactive measures to **mitigate losses** and **manage credit risk** more effectively.

The project follows a comprehensive workflow, including **data preprocessing**, **exploratory data analysis (EDA)**, **advanced feature engineering**, and a **comparative evaluation** of multiple classification models to select the best performer.

---

## 💾 Dataset

The project utilizes two datasets:

- `train_dataset_final1.csv`: Used for training and evaluating the models.  
- `validate_dataset_final.csv`: An unseen dataset used to generate the final predictions.

The data contains customer information such as demographics, credit limit, historical payment status, bill amounts, and payment amounts.

---

## ⚙️ Project Workflow

### 1. Data Preprocessing

- **Data Loading & Cleaning**: The initial dataset was loaded, and the non-predictive `Customer_ID` column was dropped.  
- **Handling Missing Values**: Missing values in the `age` column were imputed using the median to maintain the original distribution.  
- **Categorical Data Cleaning**: Inconsistent values in `marriage (0)` and `education (0, 5, 6)` were mapped to existing, logical categories.  
- **One-Hot Encoding**: Categorical features like `sex`, `marriage`, and `education` were converted into a numerical format suitable for modeling.  
- **Outlier Treatment**: The **IQR method** was applied to cap extreme values in numerical columns like `LIMIT_BAL`, `age`, and all `Bill_amt` and `pay_amt` columns. This prevents outliers from disproportionately influencing the model.

---

### 2. Feature Engineering

To enhance the model's predictive power, several new features were engineered:

- **age_group**: Customers were categorized into `adult`, `senior`, and `old` age groups.  
- **credit_utilization**: Calculated as the ratio of average bill amount to the credit limit (`AVG_Bill_amt / LIMIT_BAL`), this measures how much of the available credit a customer is using.  
- **max_delinquency_streak**: A powerful feature that calculates the longest consecutive number of months a customer was delinquent on payments. This captures patterns of chronic late payments.

---

### 3. Exploratory Data Analysis (EDA)

Visualizations were created to uncover relationships between features and the likelihood of default:

- **Delinquency Streak**: A strong positive correlation was found between the `max_delinquency_streak` and the default rate.  
- **Gender**: One gender showed a higher probability of default.  
- **Age**: A KDE plot revealed that customers aged **26–40** are less likely to default compared to others.

---

### 4. Model Training and Evaluation

- **Handling Class Imbalance**: Since the data was imbalanced, **SMOTE** (Synthetic Minority Over-sampling Technique) was used to balance the classes.  
- **Feature Scaling**: **StandardScaler** was applied to normalize numerical features.  
- **Model Comparison**: Four classification models were trained and evaluated:

  1. Logistic Regression  
  2. MLP Classifier (Neural Network)  
  3. Random Forest Classifier  
  4. XGBoost Classifier  

- **Custom Evaluation**: The **F2-score** was used as the primary metric since it emphasizes recall (catching more defaulters).  
  Each model’s probability threshold was tuned for the optimal F2-score.

| Model | Optimal Threshold | Accuracy | F1-Score | F2-Score |
|:------|:-----------------:|:---------:|:---------:|:---------:|
| Logistic Regression | 0.25 | 0.463 | 0.376 | 0.567 |
| MLP Classifier | 0.15 | 0.505 | 0.399 | 0.591 |
| Random Forest | 0.25 | 0.584 | 0.434 | 0.613 |
| XGBoost | 0.10 | 0.553 | 0.414 | 0.593 |

**Best Model:** ✅ Random Forest (Highest F2-score)

---

### 5. Final Prediction

The entire preprocessing and feature engineering pipeline was replicated on `validate_dataset_final.csv`.  
The **Random Forest model**, with an optimal threshold of **0.25**, was used to predict `next_month_default` for the validation dataset.  
Final predictions were saved to:

```
submission_<23124010>.csv
```

---

## 🚀 How to Run This Project

### Dependencies

Install the required libraries:

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost seaborn matplotlib
```

### Execution

Run the Python script or Jupyter notebook **step-by-step** to reproduce the results.  
The script will handle all stages — **data loading**, **cleaning**, **feature engineering**, **model training**, **evaluation**, and **final predictions**.

---

## 🧠 Key Takeaways

- **Delinquency streak** is the most predictive indicator of default.  
- Proper **feature engineering** and **class balancing** drastically improved recall.  
- **Random Forest** provided the best trade-off between interpretability and predictive power.

---

## 🧑‍💻 Author

Developed as part of a **credit risk prediction** project leveraging advanced machine learning and statistical techniques.

---
