import argparse
import os
import pickle
import mlflow
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def runn():
    a = mlflow.get_tracking_uri()
    b = mlflow.list_experiments()
    print(f"trackink_ui: {a}, list_experiment: {b}")


def run(data_path):

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.autolog(disable=False)

    with mlflow.start_run():

        
        
        mlflow.set_tag("model", "RandomForestRegressorModel")
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_valid)

        rmse = mean_squared_error(y_valid, y_pred, squared=False)

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="the location where the processed NYC taxi trip data was saved."
    )
    args = parser.parse_args()
    #runn()
    run(args.data_path)