# https://www.kaggle.com/hellbuoy/carprice-prediction-mlr-rfe-vif

from flask import Flask, request, jsonify
from sklearn.externals import joblib
import flask
import json
import requests
import traceback
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# API definition

test_set = None

app = Flask(__name__)

@app.route('/')
def main():
    return "Car Price Prediction"

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if flask.request.method == 'POST':
        if gb:
            try:
                # print('here')
                test_json = request.json
                # print(test_json)
                all = pd.DataFrame(test_json)
                test_ = pd.get_dummies(pd.DataFrame(test_json))
                test = test_.reindex(columns=model_columns, fill_value=0)
        
                # Normalizing the columns
                scaler = StandardScaler()
                sig_col = ['enginesize', 'horsepower']
                test[sig_col] = scaler.fit_transform(test[sig_col])
                # global pred
                pred = list(gb.predict(test))
                car_test = pd.read_csv('test.csv')
                
                test['price'] = pred
                test.update(all)
                car_test = pd.concat([car_test, test])
                car_test = car_test.drop_duplicates(keep='first')
                car_test.to_csv('test.csv', index=0)


                return jsonify({'prediction': str(pred), 'data': str(test)})

            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            print('No model available to use')
    if flask.request.method == 'GET':
        test_json = json.dumps({"enginesize": [118], "horsepower":[150], "carbody": ["hardtop"], "Make":['alfa-romero']})
        print(test_json)
        all = pd.read_json(test_json)
        test_ = pd.get_dummies(pd.read_json(test_json))
        test = test_.reindex(columns=model_columns, fill_value=0)

        # Normalizing the columns
        scaler = StandardScaler()
        sig_col = ['enginesize', 'horsepower']
        test[sig_col] = scaler.fit_transform(test[sig_col])
        pred = list(gb.predict(test))
        car_test = pd.read_csv('test.csv')
        
        test['Price'] = pred
        test.update(all)
        car_test = pd.concat([car_test, test])
        car_test = car_test.drop_duplicates(keep='first')
        car_test.to_csv('test.csv', index=0)

        return jsonify({'prediction': str(pred), 'data': str(test)})


@app.route('/train', methods=['GET', 'POST'])
def model():
    if flask.request.method == 'GET':
        # print('here')
        df = pd.read_csv('cars_selected.csv')
        # print(df.head(2))
        cat_cols = df.select_dtypes(include=['object']).columns
        dummies = pd.get_dummies(df[cat_cols], drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df.drop(cat_cols, axis=1, inplace=True)
        
        # Normalizing the columns
        scaler = StandardScaler()
        sig_col = ['enginesize', 'horsepower']
        df[sig_col] = scaler.fit_transform(df[sig_col])
        
        y = df.pop('price')
        X = df
        
        # Cross Validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Training #
        gb = GradientBoostingRegressor()
        gb.fit(X_train, y_train)
        
        # Saving model
        joblib.dump(gb, 'model.pkl')
        
        # Saving model columns
        cols = list(X_train.columns)
        joblib.dump(cols, 'model_columns.pkl')

        if test_set:
            predictions = gb.predict(test_set)
            return jsonify({"predictions": str(predictions)})
        else:
            print('model trained, testing model on validation data...')
            pred = gb.predict(X_test)
            r2score = r2_score(y_test, pred)
            print(f"Validation set r2 score: {r2score}")
            
            return jsonify({"message": "Training complete!!!", "predictions": str(pred[:3]), "r2_score": str(r2score)})

@app.route('/update_model', methods=['GET', 'POST'])
def update():
    df = pd.read_csv('test.csv')
    df['price'] = None

    return df

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 8000

    model_columns = joblib.load('model_columns.pkl')
    print('Loading model...')
    gb = joblib.load('model.pkl')
    print('Model successfully loaded!')

    app.run(port=port, debug=True)
        
