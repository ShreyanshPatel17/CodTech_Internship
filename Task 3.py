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

######  requirements.txt
Flask
numpy
pandas
scikit-learn
joblib

######  train_model.py
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# Load data
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target

# Use only 4 selected features
selected_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms']
X = df[selected_features]
y = df['PRICE']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/house_price_model.pkl")


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

</body>
</html>


