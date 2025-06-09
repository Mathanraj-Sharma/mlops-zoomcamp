from zenml import pipeline
from steps import read_data, transform_data, train_model, extract_target


@pipeline
def taxi_duration_pipeline(year: int, month: int):
    df_train = read_data(year, month)
    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_data(next_year, next_month)
    
    X_train, dv = transform_data(df_train)
    X_val, _ = transform_data(df_val, dv=dv)
    
    y_train = extract_target(df_train)
    y_val = extract_target(df_val)
    
    train_model(X_train, y_train, X_val, y_val, dv)
