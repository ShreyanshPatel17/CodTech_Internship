# Import Required Libraries
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv('Personal_Finance_and_Spendings.csv')
df.head()

# Separate features and target
X = df.drop("Rent", axis=1)
y = df["Rent"]

# Identify numerical and categorical features
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print("Numerical columns:", num_cols)
print("Categorical columns:", cat_cols)

# Numeric pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Full column transformer
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Fit and transform the data
X_prepared = full_pipeline.fit_transform(X)

print("Shape after preprocessing:", X_prepared.shape)

# Combine with target variable
X_df = pd.DataFrame(X_reduced)
y_df = pd.DataFrame(y, columns=["Rent"])
processed_df = pd.concat([X_df, y_df], axis=1)

# Output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Save to CSV
processed_df.to_csv(os.path.join(output_dir, "processed_data.csv"), index=False)

print("Processed data saved to output/processed_data.csv")
