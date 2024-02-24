import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
from tkinter import *
app = Flask(__name__)

model = joblib.load("Students_mark_predictor_model.pk1")

df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global df
    
    input_data = {
        'full_name': request.form['full_name'],
        'email': request.form['email'],
        'roll_number': request.form['roll_number'],
        'class': request.form['class'],
        'marks_percentage': float(request.form['marks_percentage']),
        'study_hours': int(request.form['study_hours'])
    }
    
    if input_data['study_hours'] < 0 or input_data['study_hours'] > 24:
        return render_template('index.html', prediction_text='Please enter valid hours between 1 to 24 if you live on the Earth')
    
    output = model.predict([[input_data['study_hours']]])[0][0].round(2)

    df = pd.concat([df, pd.DataFrame(input_data, index=[0])], ignore_index=True)
    df['Predicted_Marks'] = output
    df.to_csv('smp_data_from_app.csv', index=False)

    return render_template('index.html', Text_color='orange',prediction_text=f'Predicted marks for {input_data["full_name"]} will get {output}% marks, when he do study {input_data["study_hours"]} hours per day')

if __name__ == "__main__":
    app.run(debug=True)

