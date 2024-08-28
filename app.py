from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression


app_name = "Predicting Medical Charges"
app = Flask(app_name)

def model_building():
    data = pd.read_excel("/Users/salihburakgurhan/DataGlacier/Week-5 Flask App/data-glacier-Flask-app/medical-charges-dataset.xlsx")
    data = pd.get_dummies(data)
    X = data.drop("charges", axis=1)
    y = data["charges"]

    model = LinearRegression().fit(X,y)

    return model


@app.route('/predict', methods=['POST'])
def predict(model):
    age = int(request.form.get('age'))
    sex = request.form.get('sex')
    bmi = float(request.form.get('bmi'))
    children = int(request.form.get('children'))
    smoker = request.form.get('smoker')
    region = request.form.get('region') 

    features = pd.DataFrame({'age': [age], 'sex': [sex], 'bmi': [bmi],
                             'children': [children], 'smoker': [smoker], 'region': [region]})

    prediction = model.predict(features)[0]

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    model_building()
    app.run(debug=True)