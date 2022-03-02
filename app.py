from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from werkzeug.utils import secure_filename

app = Flask(__name__)
gbm = joblib.load('Gradient_Boosting_Model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    output = []
    if request.method == 'POST':
        input_df = pd.read_csv(request.files.get('file'))
        output = gbm.predict(input_df)
    
    return render_template('index.html', 
    prediction_text=f'The goal value for given data is: {output}')

if __name__ == "__main__":
    app.run(debug=True)
