from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

car = pd.read_csv("Cleaned Car.csv")
with open("LRm.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = sorted(car['fuel_type'].unique())

    return render_template("index.html", companies=companies, car_models=car_models, years=year, fuel_type=fuel_type)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get("SM")
    year = int(request.form.get('SY'))
    fuel_type = request.form.get('SFT')
    kms_driven = int(request.form.get('KMS'))

    data = pd.DataFrame([[company, car_model, year, fuel_type, kms_driven]],
                        columns=['company', 'name', 'year', 'fuel_type', 'kms_driven'])





    prediction = model.predict(data)
    rounded_prediction = round(prediction[0], 3)  # Round to three decimal places
    print(rounded_prediction)


    return str(rounded_prediction)

if __name__ == "__main__":
    app.run(debug=True)
