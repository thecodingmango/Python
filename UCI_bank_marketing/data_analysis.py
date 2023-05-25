# Importing necessary libraries
import pandas as pd
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pre_analysis import histogram_plot, line_plot, scatter_plot

# Read the data from the csv
banking_data = pd.read_csv('./data/bank-additional-full.csv', sep=';')

"""
# Basic data description
"""
desc_data = banking_data.describe()
for column in banking_data:
    print(f'Column_name: {column} \n'
          f'Datatype: {banking_data[column].dtypes}\n'
          f'Unique Data: {banking_data[column].unique()}\n'
          f'------------------------------------------------------\n')

# Split the data into train and test set
x = banking_data.loc[:, banking_data.columns != 'y']
y = banking_data['y']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# Using Min-Max Scaler to transform data
def min_max_scaler(data_frame, columns):
    """
    Uses the Scikit learn min max scaler to scale data
    """
    data_transformer = MinMaxScaler()

    data_frame.loc[:, columns] = data_transformer.fit_transform(data_frame.loc[:, columns])

    return data_frame


column_list = ['age', 'campaign', 'pdays', 'emp.var.rate', 'cons.price.idx', 'euribor3m', 'nr.employed']
x_train = min_max_scaler(x_train, column_list)

