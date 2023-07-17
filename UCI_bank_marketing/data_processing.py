"""
This file will help to process the data before using it for logistic regression
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class DataProcessing:

    def __init__(self):

        self.bank_data = pd.read_csv('data/bank-additional-full.csv', sep=";")
        for column in self.bank_data:
            print(f'Column_name: {column} \n'
                  f'Datatype: {self.bank_data[column].dtypes}\n'
                  f'Unique Data: {self.bank_data[column].unique()}\n'
                  f'------------------------------------------------------\n')

    def rm_special_char(self):

        self.bank_data = self.bank_data.replace(r'\.$', '', regex=True)
        self.bank_data = self.bank_data.replace(r'[^\w\s]', '_', regex=True)

        return self.bank_data



    '''   def categorical_to_numeric(self):
        """
        Given a dataframe categories, Replaces all the binary categorical variables to 0 and 1
        """

        numeric_encoder = OneHotEncoder(handle_unknown='ignore')
        numeric_encoder.fit(self.bank_data)
    '''







# bank_data = pd.read_csv('data/bank-additional-full.csv', sep=";")
data = DataProcessing()
print(data.rm_special_char())

# data.categorical_to_numeric()
# numeric_encoder = OneHotEncoder(handle_unknown='ignore')
# transformed = numeric_encoder.fit_transform(bank_data['job']).toarray()
# numeric_encoder.get_feature_names_out(['job'])
