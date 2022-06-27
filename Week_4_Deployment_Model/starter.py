#!/usr/bin/env python
# coding: utf-8
#get_ipython().system('pip freeze | grep scikit-learn')

import pickle
import pandas as pd
import sys
from flask import Flask,  request, jsonify



def load_model_dict(model):
    with open(f'{model}', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    return dv, lr


def read_data(filename):
    categorical = ['PUlocationID', 'DOlocationID']
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df





def run(year, month, dv, lr):
   
    categorical = ['PUlocationID', 'DOlocationID']
    df = read_data(f'fhv_tripdata_{year:04d}-{month:02d}.parquet')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['prediction'] = y_pred
    df_result.to_parquet(
        f'{year:04d}_{month:02d}_prediction',
        engine='pyarrow',
        compression=None,
        index=False
    )

    return df_result['prediction'].mean()


app = Flask('duration-prediction')



@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    year = ride['year']
    month = ride['month']
    model = ride['model']

    dv, lr = load_model_dict(model)

    value_mean = run(year=year, month=month, dv=dv, lr=lr)

    result = {
        'mean_of_prediction': value_mean
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)




# value of mean : 16.29