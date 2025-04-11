import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

df = pd.read_csv("car data.csv")
print("Columns:", df.columns)   # 👈 add this line to see actual column names

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# Load the dataset
df = pd.read_csv("car data.csv")

# Drop unnecessary columns
df.drop(['Unnamed: 0', 'car_name', 'brand', 'model'], axis=1, inplace=True)

# Encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"Model R² Score: {score:.2f}")

# Save the model
joblib.dump(model, "model.pkl")
