import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

def telco_prep():
    df = get_telco_data()
    df = df[['monthly_charges','tenure', 'churn','internet_service_type_id']]
    df = df.loc[:,~df.columns.duplicated()] 
    df.internet_service_type_id = df.internet_service_type_id.replace(1,"DSL")
    df.internet_service_type_id = df.internet_service_type_id.replace(2,"fiber")
    df.internet_service_type_id = df.internet_service_type_id.replace(3,"none")
    df = df.rename(columns={'internet_service_type_id': 'internet_service'})
    train_and_validate, test = train_test_split(df, test_size =.15, random_state=123)
    train, validate = train_test_split(train_and_validate, test_size=.2, random_state=123)
    return train, validate, test

def plot_varibles_pairs(df):     
    g = sns.PairGrid(df)    
    g.map_diag(sns.distplot)    
    g.map_offdiag(sns.regplot, scatter_kws={"color": "dodgerblue"}, line_kws={"color": "orange"})

def month_to_years(df):
    df['tenure_years'] = round(df['monthly_charges']/12,2)

def plot_categorical_and_continuous_vars(df, cat, cont):
    sns.boxplot(data=df,y=cont,x=cat)
    sns.swarmplot(data=df,y=cont,x=cat)   
    sns.barplot(data=df, y=cont, x=cat)