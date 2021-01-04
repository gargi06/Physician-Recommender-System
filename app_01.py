import numpy as np
import pandas as pd
from flask import Flask, render_template, request, url_for, redirect
import pickle
import final_system_prediction_01
import class_final_02
import doctor_02
import os
app = Flask(__name__)
model2 = pickle.load(open('feature_final_system.pkl', 'rb'))
model1 = pickle.load(open('feature.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():

   ######## MODEL 1 FOR PREDICTING CLASS  ################
    if request.method == 'POST':
        str_symptoms = str(request.form['symptoms'])
        prediction = class_final_02.make_prediction(str_symptoms)
        if(prediction == 1):
            prediction = "High Risk Disease"
        else:
            prediction = "Low Risk Disease"

    ####### MODEL 2 FOR PREDICTING SYSTEM ##########
    if request.method == 'POST':
        str_symptoms_new = str(request.form['symptoms'])
        syst_prediction = final_system_prediction_01.result_system(
            str_symptoms_new)
        temp = []
        dictlist = []
        for key, value in syst_prediction.items():
            temp = [key, value]
            dictlist.append(temp)

        var1, var2, var3 = [dictlist[i] for i in (0, 1, 2)]

        listToStr1 = ' '.join(
            [str(var1[0])+"% probability that it is "+str(var1[1])])
        listToStr2 = ' '.join(
            [str(var2[0])+"% probability that it is "+str(var2[1])])
        listToStr3 = ' '.join(
            [str(var3[0])+"% probability that it is "+str(var3[1])])

        # Using pkl to pass var1 of predict function to results function

        # Replace this with path of your folder
        if os.path.exists("C:/Users/Praduemna Gore/FlaskFinal/var00.txt"):
            os.remove("C:/Users/Praduemna Gore/FlaskFinal/var00.txt")

        fpp = open("C:/Users/Praduemna Gore/FlaskFinal/var00.txt", "w")

        with open("C:/Users/Praduemna Gore/FlaskFinal/var00.txt", "wb") as fp:
            pickle.dump(var1, fp)

    return render_template('index.html', prediction_text='The class of disease is: {}'.format(prediction), prediction_text_2='{}'.format(listToStr1), prediction_text_3='{}'.format(listToStr2), prediction_text_4='{}'.format(listToStr3))

# Function for Displaying  Doctor Recommendations
@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'GET':
        return render_template('results.html', data=None)
    elif request.method == 'POST':
        str_location = str(request.form['location']).title()

        # Unpickling ####### Replace this with Path of folder
        with open("C:/Users/Praduemna Gore/FlaskFinal/var00.txt", "rb") as fp:
            var = pickle.load(fp)

        str_system = str(var[1]).title()

        names, location, contacts = doctor_02.get_doctors(
            str_system, str_location)

    return render_template("results.html", names=names, contacts=contacts, location=location, system='{}'.format(str_system))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
