from flask import Flask, render_template, request, flash
import numpy as np
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)
clf = pickle.load(open('models/final_pipeline.pickle', 'rb'))


@app.route('/',methods= ['GET', 'POST'])
def toPredict():
    return render_template("index.html")

@app.route('/result',methods= ['POST'])
def result():

    data = request.form['tfTest']
    
    # Make a prediction
    prediction = clf.predict([data])

    # Determine the result message
    result_message = 'Not Fake' if prediction == 1 else 'Fake'

    return render_template('results.html', result=result_message)

#main driver
if __name__ == '__main__':
    app.run()