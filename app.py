from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Model file paths
model_files = {
    "Random Forest": "random_forest.pkl",
    "Linear Regression": "linear_regression.pkl",
    "Gradient Boosting": "gradient_boosting.pkl"
}

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Collect input
            vehicle_age = int(request.form['vehicle_age'])
            km_driven = int(request.form['km_driven'])
            mileage = float(request.form['mileage'])
            engine = int(request.form['engine'])
            max_power = float(request.form['max_power'])
            seats = int(request.form['seats'])
            fuel_type = request.form['fuel_type']
            seller_type = request.form['seller_type']
            transmission_type = request.form['transmission_type']
            selected_model_name = request.form['model_type']

            # Prepare feature dictionary
            data = {
                'vehicle_age': vehicle_age,
                'km_driven': km_driven,
                'mileage': mileage,
                'engine': engine,
                'max_power': max_power,
                'seats': seats,

                # Fuel types
                'fuel_type_Diesel': 1 if fuel_type == 'Diesel' else 0,
                'fuel_type_Petrol': 1 if fuel_type == 'Petrol' else 0,
                'fuel_type_CNG': 0,
                'fuel_type_LPG': 0,
                'fuel_type_Electric': 0,

                # Seller types
                'seller_type_Individual': 1 if seller_type == 'Individual' else 0,
                'seller_type_Dealer': 1 if seller_type == 'Dealer' else 0,
                'seller_type_Trustmark Dealer': 0,

                # Transmission types
                'transmission_type_Manual': 1 if transmission_type == 'Manual' else 0,
                'transmission_type_Automatic': 1 if transmission_type == 'Automatic' else 0
            }

            input_df = pd.DataFrame([data])

            # Load model
            model_path = model_files[selected_model_name]
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # Load feature order
            with open(model_path.replace(".pkl", "_features.pkl"), 'rb') as f:
                feature_order = pickle.load(f)

            # Reorder input to match training
            input_df = input_df[feature_order]

            # Predict
            prediction = model.predict(input_df)[0]
            predicted_price = round(prediction / 100000, 2)  # Lakhs

            return render_template('index.html', prediction=predicted_price, model_used=selected_model_name)

        except Exception as e:
            return render_template('index.html', prediction=None, model_used=None, error=str(e))

    return render_template('index.html', prediction=None, model_used=None)

if __name__ == "__main__":
    app.run(debug=True)
