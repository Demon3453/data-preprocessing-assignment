
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. LOAD RAW DATASET
# ============================================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week',
    'native_country', 'income'
]

df = pd.read_csv(
    url,
    header=None,
    names=columns,
    skipinitialspace=True
)

original_shape = df.shape

print("Original dataset shape:", original_shape)

# Save raw dataset
df.to_csv("adult_raw.csv", index=False)


# ============================================================
# 2. CREATE COPY FOR PREPROCESSING
# ============================================================

cleaned_df = df.copy()


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

# Convert '?' into missing values
cleaned_df = cleaned_df.replace("?", pd.NA)

# Fill missing categorical values with mode
categorical_missing = [
    'workclass',
    'occupation',
    'native_country'
]

for col in categorical_missing:
    cleaned_df[col] = cleaned_df[col].fillna(
        cleaned_df[col].mode()[0]
    )

print("\nMissing values after treatment:")
print(cleaned_df.isnull().sum().sum())


# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

print("\nDuplicates before removal:",
      cleaned_df.duplicated().sum())

cleaned_df = cleaned_df.drop_duplicates()

print("Duplicates after removal:",
      cleaned_df.duplicated().sum())

print("Shape after duplicate removal:",
      cleaned_df.shape)


# ============================================================
# 5. CORRECT DATA TYPES
# ============================================================

numeric_columns = [
    'age',
    'fnlwgt',
    'education_num',
    'capital_gain',
    'capital_loss',
    'hours_per_week'
]

for col in numeric_columns:
    cleaned_df[col] = pd.to_numeric(
        cleaned_df[col],
        errors='coerce'
    )


# ============================================================
# 6. DETECT OUTLIERS USING IQR
# ============================================================

print("\nOutliers detected:")

for col in numeric_columns:

    Q1 = cleaned_df[col].quantile(0.25)
    Q3 = cleaned_df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (
        (cleaned_df[col] < lower_bound) |
        (cleaned_df[col] > upper_bound)
    ).sum()

    print(f"{col}: {outliers}")


# ============================================================
# 7. TREAT OUTLIERS USING IQR CAPPING
# ============================================================

for col in numeric_columns:

    Q1 = cleaned_df[col].quantile(0.25)
    Q3 = cleaned_df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    cleaned_df[col] = cleaned_df[col].clip(
        lower=lower_bound,
        upper=upper_bound
    )

print("\nOutliers treated using IQR capping.")


# ============================================================
# 8. SEPARATE FEATURES AND TARGET
# ============================================================

X = cleaned_df.drop('income', axis=1)
y = cleaned_df['income']


# ============================================================
# 9. ENCODE TARGET VARIABLE
# ============================================================

y = y.str.strip()

y = y.map({
    '<=50K': 0,
    '>50K': 1
})


# ============================================================
# 10. ONE-HOT ENCODE CATEGORICAL FEATURES
# ============================================================

X = pd.get_dummies(
    X,
    drop_first=True
)

# Convert Boolean columns to integers
X = X.astype(int)

print("\nShape after encoding:", X.shape)


# ============================================================
# 11. STANDARDIZE NUMERICAL FEATURES
# ============================================================

scaler = StandardScaler()

X[numeric_columns] = scaler.fit_transform(
    X[numeric_columns]
)

print("\nNumerical features standardized.")


# ============================================================
# 12. COMBINE FEATURES AND TARGET
# ============================================================

cleaned_final = X.copy()

cleaned_final['income'] = y


# ============================================================
# 13. REMOVE FINAL DUPLICATES
# ============================================================

print("\nDuplicates before final removal:",
      cleaned_final.duplicated().sum())

cleaned_final = cleaned_final.drop_duplicates()

print("Duplicates after final removal:",
      cleaned_final.duplicated().sum())


# ============================================================
# 14. FINAL DATASET CHECK
# ============================================================

print("\n========== FINAL DATASET CHECK ==========")

print("Rows:", cleaned_final.shape[0])
print("Columns:", cleaned_final.shape[1])

print(
    "Missing values:",
    cleaned_final.isnull().sum().sum()
)

print(
    "Duplicate rows:",
    cleaned_final.duplicated().sum()
)

print("\nData types:")
print(cleaned_final.dtypes.value_counts())


# ============================================================
# 15. SAVE FINAL CLEANED DATASET
# ============================================================

cleaned_final.to_csv(
    "adult_cleaned.csv",
    index=False
)

print("\nadult_cleaned.csv saved successfully!")
