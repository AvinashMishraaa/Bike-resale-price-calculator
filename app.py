from flask import Flask, render_template_string, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the model and scalers
model = joblib.load('bike_model.pkl')
scaler_x = joblib.load('scaler_x.pkl')
scaler_y = joblib.load('scaler_y.pkl')

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Bike Resale Price Calculator</title>
    <style>
        body {
            font-family: Arial;
            background-color: #f0f0f0;
            text-align: center;
            margin: 0;
            padding: 0;
        }
        .nav {
            background-color: #222;
            overflow: hidden;
            padding: 14px 0;
        }
        .nav a {
            color: #00ffcc;
            text-decoration: none;
            margin: 0 20px;
            font-weight: bold;
        }
        .container {
            background-color: #333;
            color: white;
            margin: 60px auto;
            padding: 30px;
            border-radius: 10px;
            width: 500px;
        }
        input[type="text"], button {
            width: 90%;
            padding: 12px;
            margin: 10px 0;
            border-radius: 5px;
            border: none;
            font-size: 16px;
        }
        button {
            background-color: #00ffcc;
            color: black;
            cursor: pointer;
        }
        .result {
            font-size: 22px;
            color: #00ffcc;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="nav">
        <a href="/">Bike Resale Price Calculator</a>
    </div>
    <div class="container">
        <h1>Bike Resale Price Calculator</h1>
        <form method="POST">
            <input type="text" name="f0" placeholder="Avg Daily Distance (km)" required>
            <input type="text" name="f1" placeholder="Price (INR)" required>
            <input type="text" name="f2" placeholder="Year of Manufacture" required>
            <input type="text" name="f3" placeholder="Engine Capacity (cc)" required>
            <input type="text" name="f4" placeholder="Mileage   (km/l)" required>
            <input type="text" name="f5" placeholder="Registration Year" required>
            <button type="submit">Predict</button>
        </form>
        {% if result %}
            <div class="result">Estimated Resale Price: ₹ {{ result }}</div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        try:
            features = [float(request.form[f'f{i}']) for i in range(6)]
            features_scaled = scaler_x.transform([features])
            prediction_scaled = model.predict(features_scaled)
            prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
            result = f"{prediction:.2f}"
        except Exception as e:
            result = f"Error: {str(e)}"
    return render_template_string(HTML, result=result)

if __name__ == '__main__':
    app.run(debug=True)
