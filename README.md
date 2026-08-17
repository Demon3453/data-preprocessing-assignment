# data-preprocessing-assignment
UCI Machine Learning Repository – Adult Dataset (Census Income Dataset)

The official UCI page describes it as a classification dataset for predicting whether annual income exceeds $50K
Before preprocessing:
Rows: 32,561
Columns: 15

Problems identified:

Missing values represented by ?
Duplicate records
Numerical outliers
Categorical variables
Numerical variables requiring standardization
Inconsistent/raw data types

Preprocessing techniques:

Missing-value detection
Mode imputation
Duplicate removal
IQR outlier detection
IQR outlier capping
Data-type correction
One-hot encoding
Target encoding

| Stage                               |       Rows | Columns |
| ----------------------------------- | ---------: | ------: |
| Raw dataset                         |     32,561 |      15 |
| After duplicate removal             |     32,537 |      15 |
| After one-hot encoding              |     32,537 |     100 |
| Final before duplicate verification |     32,537 |     101 |
| **Final cleaned dataset**           | **32,510** | **101** |

| Stage                               |       Rows | Columns |
| ----------------------------------- | ---------: | ------: |
| Raw dataset                         |     32,561 |      15 |
| After duplicate removal             |     32,537 |      15 |
| After one-hot encoding              |     32,537 |     100 |
| Final before duplicate verification |     32,537 |     101 |
| **Final cleaned dataset**           | **32,510** | **101** |

Standardization
Column naming/cleaning
