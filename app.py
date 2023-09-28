from flask import Flask, render_template, request, flash, redirect
import numpy as np
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
clf = pickle.load(open('models/final_pipeline.pickle', 'rb'))
result_message = None

@app.route('/',methods= ['GET','POST'])
def main():
    return render_template("index.html", result=None)

@app.route('/predict',methods= ['POST'])
def toPredict():
    global result_message
    
    if request.method == 'POST':
        data = request.form['tfTest']
        prediction = clf.predict([data])
        result_message = 'Not Fake' if prediction == 1 else 'Fake'
    
    return render_template("index.html", result=result_message)

if __name__ == '__main__':
    app.run(debug=True)