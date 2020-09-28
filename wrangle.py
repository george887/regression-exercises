import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def wrangle_telco(df):
    df = df[df['two_year'] == 1]
    df = df[['customer_id', 'monthly_charges', 'tenure', 'total_charges']]
    return df
