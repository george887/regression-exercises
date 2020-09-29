import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

def wrangle_telco_data(df):
    df = df[df['two_year'] == 1]
    df = df[['customer_id', 'monthly_charges', 'tenure', 'total_charges']]
    return df

def add_scaled_columns(train, validate, test, scaler, columns_to_scale):
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    scaler.fit(train[columns_to_scale])

    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index),
    ], axis=1)
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index),
    ], axis=1)

    return train, validate, test

def scale_telco_data(train, test, validate):
    train, validate, test = add_scaled_columns(
        train,
        test,
        validate,
        scaler=sklearn.preprocessing.MinMaxScaler(),
        columns_to_scale=['total_charges', 'monthly_charges', 'tenure'],
    )
    return train, validate, test

def wrangle_grades():
    grades = pd.read_csv("student_grades.csv")
    grades.drop(columns="student_id", inplace=True)
    grades.replace(r"^\s*$", np.nan, regex=True, inplace=True)
    df = grades.dropna().astype("int")
    return df