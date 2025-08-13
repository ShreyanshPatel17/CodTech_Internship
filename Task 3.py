######  requirements.txt
Flask
numpy
pandas
seaborn
scikit-learn
joblib



######  app.py
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("model/house_price_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms']
        input_features = [float(request.form[feature]) for feature in features]
        prediction = model.predict([np.array(input_features)])
        return render_template("index.html", prediction_text=f'Estimated House Price: ${round(prediction[0]*100, 2)}')

if __name__ == '__main__':
    app.run(debug=True)



######  train_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os
from scipy import stats

# Create plots directory
os.makedirs("static/plots", exist_ok=True)

# ---------------------------
# 1. Load Data
# ---------------------------
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target

print("Initial Data Shape:", df.shape)

# ---------------------------
# 2. Data Cleaning
# ---------------------------
print("\nMissing values in dataset:")
print(df.isnull().sum())

# Fill missing values if any
df.fillna(df.median(), inplace=True)

# Remove outliers
z_scores = np.abs(stats.zscore(df))
df = df[(z_scores < 3).all(axis=1)]
print("Shape after removing outliers:", df.shape)

# ---------------------------
# 3. Data Visualization
# ---------------------------
sns.set_style("whitegrid")

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("static/plots/correlation_heatmap.png")
plt.close()

# Distribution of target variable
plt.figure(figsize=(8, 5))
sns.histplot(df['PRICE'], bins=50, kde=True)
plt.title("House Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.savefig("static/plots/price_distribution.png")
plt.close()

# Pairplot
selected_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'PRICE']
sns.pairplot(df[selected_features])
plt.savefig("static/plots/feature_pairplot.png")
plt.close()

print("✅ Plots saved in static/plots/")

# ---------------------------
# 4. Model Training
# ---------------------------
X = df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms']]
y = df['PRICE']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/house_price_model.pkl")
print("✅ Model saved to model/house_price_model.pkl")




#####  index.html
<!DOCTYPE html>
<html>
<head>
    <title>California House Price Predictor</title>
</head>
<body>
    <h2>California House Price Prediction</h2>
    <form action="/predict" method="post">
       <input type="text" name="MedInc" placeholder="Median Income"><br>
       <input type="text" name="HouseAge" placeholder="House Age"><br>
       <input type="text" name="AveRooms" placeholder="Average Rooms"><br>
       <input type="text" name="AveBedrms" placeholder="Average Bedrooms"><br>
       <input type="submit" value="Predict">
    </form>
    <h3>{{ prediction_text }}</h3>

    <hr>
    <h2>Data Visualizations</h2>
    <div>
        <h4>Correlation Heatmap</h4>
        <img src="{{ url_for('static', filename='plots/correlation_heatmap.png') }}" width="600">
    </div>
    <div>
        <h4>Price Distribution</h4>
        <img src="{{ url_for('static', filename='plots/price_distribution.png') }}" width="600">
    </div>
    <div>
        <h4>Feature Pairplot</h4>
        <img src="{{ url_for('static', filename='plots/feature_pairplot.png') }}" width="600">
    </div>
</body>
</html>



