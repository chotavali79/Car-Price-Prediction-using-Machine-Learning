# Car-Price-Prediction-using-Machine-Learning
Built a machine learning web app to predict used car prices based
Car Price Prediction using Machine Learning
This project is a web-based Machine Learning application that predicts the resale value of a used car based on user-input features like vehicle age, kilometers driven, engine power, fuel type, etc. It uses a trained regression model deployed via a Flask backend with a simple and interactive user interface.

🔥 Features
Predicts car price based on real-world features

User-friendly web interface built using Flask

ML model trained on real car data

Supports categorical and numerical inputs

Returns predicted price in INR Lakhs

🧠 Tech Stack / Tools Used
Category	Tools / Libraries
Languages	Python, HTML, CSS
ML Libraries	Scikit-learn, Pandas, NumPy
Web Framework	Flask, Jinja2
Model Serialization	Pickle
Version Control	Git & GitHub
Dataset	Custom preprocessed car dataset (CSV format)
🛠️ ML Models Used
Random Forest Regressor

(Optional) Comparison done with:

Linear Regression

XGBoost

Gradient Boosting

AdaBoost

Bagging Regressor

Final deployed model: Random Forest with ~0.92 R² score

🚀 How to Run Locally
bash
Copy
Edit
git clone https://github.com/chotavali79/Car-Price-Prediction-using-Machine-Learning.git
cd Car-Price-Prediction-using-Machine-Learning
pip install -r requirements.txt
python app.py
Then open your browser and go to:
http://127.0.0.1:5000

📸 UI Screenshot (Optional — you can take a screenshot and add here)
🧠 Key Concepts Demonstrated
Data preprocessing & feature engineering

Model evaluation using R² score

Categorical encoding & scaling

Model serialization with Pickle

Building an end-to-end ML product

📌 Folder Structure
kotlin
Copy
Edit
├── app.py
├── model_training.py
├── train_models.py
├── model.pkl
├── car data.csv
├── templates/
│   └── index.html


👨‍💻 Author
Shaik Chotavali


