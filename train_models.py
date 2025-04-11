import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load data
df = pd.read_csv("car data.csv")
df.drop(columns=['Unnamed: 0', 'car_name', 'brand', 'model'], inplace=True)

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['fuel_type', 'seller_type', 'transmission_type'], drop_first=False)

X = df.drop(columns='selling_price')
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'linear_regression': LinearRegression(),
    'random_forest': RandomForestRegressor(),
    'gradient_boosting': GradientBoostingRegressor()
}

# Train, save models and feature orders


from sklearn.metrics import r2_score

# Train, save models and feature orders
for name, model in models.items():
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"✅ {name} R² Score: {r2:.4f}")

    # Save model
    with open(f"{name}.pkl", 'wb') as f:
        pickle.dump(model, f)

    # Save training feature order
    with open(f"{name}_features.pkl", 'wb') as f:
        pickle.dump(X.columns.tolist(), f)

    print(f"💾 {name} model and features saved.\n")
