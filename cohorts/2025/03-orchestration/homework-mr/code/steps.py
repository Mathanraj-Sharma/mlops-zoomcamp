import mlflow.sklearn
import mlflow.sklearn
from zenml import step

import scipy
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from typing import Tuple


@step
def read_data(year: int, month: int) -> pd.DataFrame:
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    print(f"Read {len(df)} rows from {url}")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print(f"Filtered to {len(df)} rows with valid durations")

    return df


@step
def transform_data(df: pd.DataFrame, dv: DictVectorizer | None = None) -> Tuple[scipy.sparse._csr.csr_matrix, DictVectorizer]:
    features = df[['PULocationID', 'DOLocationID', 'trip_distance']].to_dict(orient='records')
    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(features)
    else:
        X = dv.transform(features)
    return X, dv


@step
def extract_target(df: pd.DataFrame) -> np.ndarray:
    return df["duration"].values


@step
def train_model(X_train, y_train, X_val, y_val, dv: DictVectorizer) -> str:
    import mlflow
    import pickle
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import root_mean_squared_error
    from pathlib import Path

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("nyc-taxi-experiment")

    mlflow.sklearn.autolog()
    
    with mlflow.start_run() as run:
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        models_folder = Path("models")
        models_folder.mkdir(exist_ok=True)
        with open(models_folder / "preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact(models_folder / "preprocessor.b", artifact_path="preprocessor")
        
        print(f"Model intercept_: {model.intercept_}")

        return run.info.run_id

