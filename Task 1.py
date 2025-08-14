# Import Required Libraries
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# ---------------------------
# Load Data
# ---------------------------
df = pd.read_csv('Personal_Finance_and_Spendings.csv')
print("Initial Data Sample:")
print(df.head())

# Separate features and target
X = df.drop("Rent", axis=1)
y = df["Rent"]

# Identify numerical and categorical features
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print("Numerical columns:", num_cols)
print("Categorical columns:", cat_cols)

# ---------------------------
# Preprocessing Pipelines
# ---------------------------

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

# ---------------------------
# Fit & Transform Data
# ---------------------------
X_prepared = full_pipeline.fit_transform(X)
print("Shape after preprocessing:", X_prepared.shape)

# ---------------------------
# Apply PCA (optional step)
# ---------------------------
# Keep 95% variance
pca = PCA(n_components=0.95, random_state=42)
X_reduced = pca.fit_transform(X_prepared)
print("Shape after PCA:", X_reduced.shape)

# ---------------------------
# Build Column Names
# ---------------------------
# Get column names after encoding
encoded_cat_cols = full_pipeline.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols)
processed_feature_names = num_cols + list(encoded_cat_cols)

# Adjust names for PCA output
pca_columns = [f"PCA_{i+1}" for i in range(X_reduced.shape[1])]

# ---------------------------
# Create Processed DataFrame
# ---------------------------
X_df = pd.DataFrame(X_reduced, columns=pca_columns)
y_df = pd.DataFrame(y, columns=["Rent"])
processed_df = pd.concat([X_df, y_df], axis=1)

# ---------------------------
# Save to CSV
# ---------------------------
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "processed_data.csv")
processed_df.to_csv(output_path, index=False)

print(f"Processed data saved to {output_path}")
