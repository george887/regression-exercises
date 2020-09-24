import pandas as pd
from env import host, user, password
import os

# establish mysql connection
def get_connection(db, user=user, host=host, password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

############# Mall Customers Data ##############
def new_mall_data():
    '''
    This function reads the mall customer data from the Codeup db into a df,
    write it to a csv file, and returns the df. 
    '''
    sql_query = 'SELECT * FROM customers'
    df = pd.read_sql(sql_query, get_connection('mall_customers'))
    df.to_csv('mall_customers_df.csv')
    return df

def get_mall_data(cached=False):
    '''
    This function reads in mall customer data from Codeup database if cached == False 
    or if cached == True reads in mall customer df from a csv file, returns df
    '''
    if cached or os.path.isfile('mall_customers_df.csv') == False:
        df = new_mall_data()
    else:
        df = pd.read_csv('mall_customers_df.csv', index_col=0)
    return df

############# Acquire from titanic ###########

# create a variable for sql query
def new_titanic_data():
    sql_query = 'SELECT * FROM passengers'
    df = pd.read_sql(sql_query, get_connection('titanic_db'))
    df.to_csv('titanic_df.csv')
    return df
def get_titanic_data(cached=False):
    '''
    This function reads in titanic data from Codeup database if cached == False
    or if cached == True reads in titanic df from a csv file, returns df
    '''
    if cached or os.path.isfile('titanic_df.csv') == False:
        df = new_titanic_data()
    else:
        df = pd.read_csv('titanic_df.csv', index_col=0)
    return df

############# Acquire from iris ###########

def new_iris_data():
    '''
    This function reads the iris data from the Codeup db into a df,
    writes it to a csv file, and returns the df.
    '''
    sql_query = """
                SELECT species_id,
                measurement_id,
                species_name,
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
                FROM measurements
                JOIN species
                USING(species_id)
                """
    df = pd.read_sql(sql_query, get_connection('iris_db'))
    df.to_csv('iris_df.csv')
    return df

def get_iris_data(cached=False):
    '''
    This function reads in iris data from Codeup database if cached == False
    or if cached == True reads in iris df from a csv file, returns df
    '''
    if cached or os.path.isfile('iris_df.csv') == False:
        df = new_iris_data()
    else:
        df = pd.read_csv('iris_df.csv', index_col=0)
    return df

############# Acquire from Telco ###########
    
    # Creating a function to access the telco data
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
        df = pd.read_sql(sql_querry, get_connection('telco_churn'))
        # Create a DF with the telco data
        df.to_csv('telco_churn.csv')
        return df