from flask import Flask, request, jsonify, abort, redirect, url_for
import pandas as pd
import xgboost as xgb
import pickle

model = pickle.load(open("500-3000noGPUtr.pickle.dat", "rb"))
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/badrequest400')
def bad_request():
    return abort(400)

@app.route('/get_predict', methods=['POST'])
def add_message():
    try:
        content = request.get_json()
        X = pd.read_json(content, typ='series').to_frame().T
        predict = model.predict(X)
        predict = {'predict' : str(predict[0])}
    except:
        return redirect(url_for('bad_request'))
    return jsonify(predict)