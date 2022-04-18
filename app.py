from flask import Flask, render_template, request, jsonify
import tflite_runtime.interpreter as tflite
import joblib
from flask_cors import CORS
import argparse

app = Flask(__name__)
CORS(app)


tflite_model_name = 'kidney_prediction_model.tflite'
tflite_model_path = tflite_model_name

interpreter = tflite.Interpreter(model_path=tflite_model_path)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scaler = joblib.load('scaler.joblib')


d = {}
d['yes'] = 1
d['no'] = 0
d['present'] = 1
d['notpresent'] = 0
d['abnormal'] = 1
d['normal'] = 0
d['good'] = 1
d['poor'] = 0

d


def model_prediction(data):
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    return pred


def predict(data):
    for i in range(len(data)):
        try:
            data[i] = float(data[i])
        except:
            data[i] = d[data[i]]
    data = scaler.transform([data])
    data = data.astype('float32')
    prob = model_prediction(data)[0][0]

    if round(prob) == 0:
        return round(prob), 1-prob
    else:
        return round(prob), prob


@app.route('/', methods=['GET', 'POST'])
def hello():
    return jsonify({'kidney': '/kidney', 'liver': '/liver'})


@app.route('/kidney', methods=['GET', 'POST'])
def kidney():
    if request.method == 'GET':
        return jsonify({'message': "This gives Kidney disease prediction"},
                       {"POST-Key for uploading data": [
                           {
                               "age": "54",
                               "blood_pressure": "70",
                               "specific_gravity": "1.005",
                               "albumin": "4",
                               "sugar": "0",
                               "red_blood_cells": "abnormal",
                               "pus_cell": "normal",
                               "pus_cell_clumps": "notpresent",
                               "bacteria": "present",
                               "blood_glucose_random": "117",
                               "blood_urea": "56",
                               "serum_creatinine": "3.8",
                               "sodium": "111",
                               "potassium": "2.5",
                               "haemoglobin": "11.2",
                               "packed_cell_volume": "32",
                               "white_blood_cell_count": "6700",
                               "red_blood_cell_count": "3.9",
                               "hypertension": "yes",
                               "diabetes_mellitus": "yes",
                               "coronary_artery_disease": "yes",
                               "appetite": "poor",
                               "pedal_edema": "no",
                               "anemia": "no"
                           }
                       ]})

    elif request.method == 'POST':

        request_data = request.get_json()
        age = request_data['age']  # int
        blood_pressure = request_data['blood_pressure']  # int
        specific_gravity = request_data['specific_gravity']  # float
        albumin = request_data['albumin']  # int
        sugar = request_data['sugar']  # int
        # 'abnormal' or 'normal'
        red_blood_cells = request_data['red_blood_cells']
        pus_cell = request_data['pus_cell']  # 'abnormal' or 'normal'
        # 'present' or 'notpresent'
        pus_cell_clumps = request_data['pus_cell_clumps']
        bacteria = request_data['bacteria']  # 'present' or 'notpresent'
        blood_glucose_random = request_data['blood_glucose_random']  # int
        blood_urea = request_data['blood_urea']  # int
        serum_creatinine = request_data['serum_creatinine']  # float
        sodium = request_data['sodium']  # int
        potassium = request_data['potassium']  # float
        haemoglobin = request_data['haemoglobin']  # float
        packed_cell_volume = request_data['packed_cell_volume']  # int
        white_blood_cell_count = request_data['white_blood_cell_count']  # int
        red_blood_cell_count = request_data['red_blood_cell_count']  # float
        hypertension = request_data['hypertension']  # 'yes' or 'no'
        diabetes_mellitus = request_data['diabetes_mellitus']  # 'yes' or 'no'
        # 'yes' or 'no'
        coronary_artery_disease = request_data['coronary_artery_disease']
        appetite = request_data['appetite']  # 'poor' or 'good'
        pedal_edema = request_data['pedal_edema']  # 'yes' or 'no'
        anemia = request_data['anemia']  # 'yes' or 'no'

        # data = [eval(i) for i in col]
        data = [age,
                blood_pressure,
                specific_gravity,
                albumin,
                sugar,
                red_blood_cells,
                pus_cell,
                pus_cell_clumps,
                bacteria,
                blood_glucose_random,
                blood_urea,
                serum_creatinine,
                sodium,
                potassium,
                haemoglobin,
                packed_cell_volume,
                white_blood_cell_count,
                red_blood_cell_count,
                hypertension,
                diabetes_mellitus,
                coronary_artery_disease,
                appetite,
                pedal_edema,
                anemia]
        print(data)
        pred, prob = predict(data)
        print(pred)
        print(prob)

        if pred == 1:
            return jsonify({'message': 'Kidney Disease Detected', 'probability': str(prob)})
        elif pred == 0:
            return jsonify({'message': 'Kidney Disease Not Detected', 'probability': str(prob)})


#############################liver#############################################

@app.route('/liver', methods=['GET', 'POST'])
def liver():
    if request.method == 'GET':
        return jsonify({'message': "This gives Liver disease prediction"},
                       {"POST-Key for uploading data": [
                           {
                               "Age": "52",                            # int
                               "Gender": "Female",                   # 'Male' or 'Female'
                               "Total_Bilirubin": "10.1",              # float
                               "Direct_Bilirubin": "5.5",              # float
                               "Alkaline_Phosphotase": "490",          # int
                               "Alamine_Aminotransferase": "60",       # int
                               "Aspartate_Aminotransferase": "68",     # int
                               "Total_Protiens": "6.8",                # float
                               "Albumin": "3.3",                       # float
                               "Albumin_and_Globulin_Ratio": "0.74"   # float
                           }
                       ]})

    elif request.method == 'POST':
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')

        def predict(data):
            if data[1] == 'Female':
                data[1] = 0
            else:
                data[1] = 1
            data = scaler.transform([data])
            pred = model.predict_proba(data)
            return pred.argmax(), pred.ravel()[pred.argmax()]

        request_data = request.get_json()
        Age = request_data['Age']                            # int
        Gender = request_data['Gender']                   # 'Male' or 'Female'
        Total_Bilirubin = request_data['Total_Bilirubin']             # float
        Direct_Bilirubin = request_data['Direct_Bilirubin']             # float
        # int
        Alkaline_Phosphotase = request_data['Alkaline_Phosphotase']
        # int
        Alamine_Aminotransferase = request_data['Alamine_Aminotransferase']
        # int
        Aspartate_Aminotransferase = request_data['Aspartate_Aminotransferase']
        Total_Protiens = request_data['Total_Protiens']                # float
        Albumin = request_data['Albumin']                       # float
        Albumin_and_Globulin_Ratio = request_data['Albumin_and_Globulin_Ratio']

        # data = [eval(i) for i in col]
        data = [
            Age,                           # int
            Gender,                   # 'Male' or 'Female'
            Total_Bilirubin,             # float
            Direct_Bilirubin,             # float
            Alkaline_Phosphotase,         # int
            Alamine_Aminotransferase,      # int
            Aspartate_Aminotransferase,   # int
            Total_Protiens,             # float
            Albumin,                     # float
            Albumin_and_Globulin_Ratio,
        ]
        pred, prob = predict(data)
        print(pred)
        print(prob)

        if pred == 1:
            return jsonify({'message': 'Liver Disease Detected', 'probability': prob})
        elif pred == 0:
            return jsonify({'message': 'Liver Disease Not Detected', 'probability': 1-prob})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()
    app.run(debug=True, port=args.port)
