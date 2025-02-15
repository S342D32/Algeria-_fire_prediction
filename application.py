from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# import ridge regressor and standardscaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            Temperature = float(request.form.get('Temperature', 0))
            RH = float(request.form.get('RH', 0))
            Ws = float(request.form.get('Ws', 0))
            Rain = float(request.form.get('Rain', 0))
            FFMC = float(request.form.get('FFMC', 0))
            DMC = float(request.form.get('DMC', 0))
            DC = float(request.form.get('DC', 0))
            ISI = float(request.form.get('ISI', 0))
            BUI = float(request.form.get('BUI', 0))
            FWI = float(request.form.get('FWI', 0))
            Classes = float(request.form.get('Classes', 0))
            Region = float(request.form.get('Region', 0))

            new_data = [[Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI, FWI, Classes, Region]]
            new_data_scaled = scaler.transform(new_data)

            result = ridge_model.predict(new_data_scaled)
            return render_template('home.html', result=result[0])
        except TypeError as e:
            return f"Error: {str(e)}", 400

    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
