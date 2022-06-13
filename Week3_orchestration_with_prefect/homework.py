import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import mlflow

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

from datetime import datetime
from dateutil.relativedelta import relativedelta
import glob

import pickle




@task
def read_data(path):
    df = pd.read_parquet(path)
    return df


@task
def get_path(date=None):

    train_path =  ""
    val_path = ""

    logger = get_run_logger()

    if date == None:

        
        """
        date_after_month = datetime.today()+ relativedelta(months=1)
        print('Today: ',datetime.today().strftime('%d/%m/%Y'))
        print('After Month:', date_after_month.strftime('%d/%m/%Y'))
        """

        
        date = datetime.today()
        date_pass_m2 = date + relativedelta(months=-2)
        date_pass_m1 = date + relativedelta(months=-1)

        train_path_date = date_pass_m2.strftime('%Y-%m')
        val_path_date = date_pass_m1.strftime('%Y-%m')

        train_path = glob.glob(f"./data/fhv_tripdata_{train_path_date}.parquet")[0]
        val_path = glob.glob(f"./data/fhv_tripdata_{val_path_date}.parquet")[0]

        logger.info(f"Path of train data {train_path}!")
        logger.info(f"Path of validation data {val_path}!")
        
        
    else:
        # train_path = date - 2 months back & val_path = date - 1 month back

        date = datetime.strptime(date, "%Y-%m-%d").date()

        date_pass_m2 = date + relativedelta(months=-2)
        date_pass_m1 = date + relativedelta(months=-1)

        train_path_date = date_pass_m2.strftime('%Y-%m')
        val_path_date = date_pass_m1.strftime('%Y-%m')

        train_path = glob.glob(f"./data/fhv_tripdata_{train_path_date}.parquet")[0]
        val_path = glob.glob(f"./data/fhv_tripdata_{val_path_date}.parquet")[0]
        logger.info(f"Path of train data {train_path}!")
        logger.info(f"Path of validation data {val_path}!")
    
    return train_path, val_path







@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    logger = get_run_logger()

    mean_duration = df.duration.mean()
    if train:
        #print(mean_duration)
        logger.info(f"The mean duration of validation {mean_duration}!")
        #log_print("The mean duration of training", mean_duration)
    else:
        print(mean_duration)
        logger.info(f"The mean duration of validation {mean_duration}!")
        #log_print("The mean duration of validation", mean_duration)
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values
    logger = get_run_logger()

    logger.info(f"The shape of X_train {X_train.shape}!")
    logger.info(f"Number of The DictVectorizer features {len(dv.feature_names_)} and the size is {dv.__sizeof__()}")
    #log_print("The shape of X_train", X_train.shape)
    #log_print("Number of The DictVectorizer features ", len(dv.feature_names_))
    

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    
    logger.info(f"The MSE of validation {mse}!")
    #log_print("The MSE of training", mse)
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger = get_run_logger()
    logger.info(f"The MSE of validation {mse}!")
    #log_print("The MSE of validation", mse)
    return




@flow(task_runner=SequentialTaskRunner())
def main(date="2021-08-15"):
    

    train_path, val_path = get_path(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    with open(f"./models/model-{date}.bin", "wb") as f_out:
            pickle.dump(lr, f_out)

    with open(f"./models/dv-{date}.b", "wb") as f_out:
            pickle.dump(dv, f_out)

    run_model(df_val_processed, categorical, dv, lr)


from prefect.orion.schemas.schedules import CronSchedule
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    flow=main,
    name="model_workflow_orchestration_schedule_cron",
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"],
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York")
)
