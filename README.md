# Task 1 — Data Cleaning & Preprocessing (Titanic Dataset)

## Overview
This task focuses on cleaning and preparing the Titanic dataset for machine learning.  
I handled missing data, encoded categorical variables, scaled numerical features, and removed outliers.

## Steps Completed
1. Loaded the dataset and checked data types, shape, and missing values.
2. Filled missing values:
   - Age → median
   - Embarked → mode
   - Fare → median
3. Converted categorical columns into numeric using one-hot encoding.
4. Scaled numerical features (Age, Fare) using StandardScaler.
5. Removed outliers using the IQR method.
6. Displayed the cleaned dataset for verification.

## Tech Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn (for outlier inspection)

## Result
A fully cleaned, encoded, and scaled dataset ready for ML model training.
