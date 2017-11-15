from flask import Flask, render_template, request
import io
import re
import base64
import flask

import pandas as pd
import markovify
import time

app = Flask(__name__)

#read model from pickle
whiteHouseBriefingsDf = pd.read_pickle('../whiteHousePressBriefings_Data/processedBriefings.pkl')
spicerCorpus = '. '.join(whiteHouseBriefingsDf[(whiteHouseBriefingsDf.dateTime > '2017-01-21') &
                      ~((whiteHouseBriefingsDf.speaker == 'SPEECH') |
                       (whiteHouseBriefingsDf.speaker == 'Q'))].paragraph)
text_model = markovify.Text(spicerCorpus)

def spicersAnswer():
    return('\n'.join([text_model.make_short_sentence(100) for i in range(4)]))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['GET','POST'])
def predict():
    results = {'predictions' : spicersAnswer()}
    time.sleep(1)

    return flask.jsonify(results)
    #return('test')



if __name__ == '__main__':
    app.run(debug=True)
