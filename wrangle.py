import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from env import host, user, password
import os

def get_db_url(db_name):
    return f"mysql+pymysql://{user}:{password}@{host}/{db_name}"

def get_telco_data():
    filename = "telco.csv"
    # if a file is found with a name that matches filename (telco.csv), the function will return the data as a dataframe
    if os.path.isfile(filename):
        return pd.read_csv(filename)
        
    else:
        sql_querry = '''
                        SELECT * FROM customers AS cust
                        JOIN contract_types AS ct ON cust.contract_type_id = ct.contract_type_id
                        JOIN internet_service_types AS i_s ON cust.internet_service_type_id = i_s.internet_service_type_id
                        JOIN payment_types AS pt ON cust.payment_type_id = pt.payment_type_id;
                    '''
        df = pd.read_sql(sql_querry, get_db_url('telco_churn'))
        # Create a DF with the telco data
        df.to_csv('telco_churn.csv')
        return df

def wrangle_telco_data():
    #df = df[df['two_year'] == 1]
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