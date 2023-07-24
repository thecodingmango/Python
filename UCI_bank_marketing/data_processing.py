"""
This file will help to process the data before using it for logistic regression
"""
import pandas as pd
from pandas.api.types import is_object_dtype
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder


class DataProcessing:

    def __init__(self):

        self.bank_data = pd.read_csv('data/bank-additional-full.csv', sep=";")
        for column in self.bank_data:
            print(f'Column_name: {column} \n'
                  f'Datatype: {self.bank_data[column].dtypes}\n'
                  f'Unique Data: {self.bank_data[column].unique()}\n'
                  f'------------------------------------------------------\n')

    def rm_special_char(self):
        """
        Removes special characters '_' & '.' from the strings
        """

        self.bank_data = self.bank_data.replace(r'\.$', '', regex=True)
        self.bank_data = self.bank_data.replace(r'[^\w\s]', '_', regex=True)

        return self.bank_data

    def categorical_encode(self):
        """
        Given a dataframe categories,
        Replaces all the binary categorical variables to 0 and 1
        One Hot encodes categorical multi-class categorical variables
        """

        # Check if column is object datatype
        for column in self.bank_data:

            if is_object_dtype(self.bank_data[column]):

                # If the number of unique classes is greater than 2, then it converts it into binary classification
                if self.bank_data[column].nunique() == 2:

                    le_encoder = LabelEncoder()

                    self.bank_data[column] = le_encoder.fit_transform(self.bank_data[column])

                # If the number of unique classes is greater than 2, then it converts to n classes
                elif self.bank_data[column].nunique() > 2:

                    oe_encoder = OneHotEncoder(handle_unknown='ignore')

                    col_name = [column + '_' + name for name in sorted(self.bank_data[column].unique())]
                    new_df = pd.DataFrame(oe_encoder.fit_transform(self.bank_data[[column]]).toarray(),
                                          columns=col_name)

                    self.bank_data = self.bank_data.drop(column, axis=1)
                    self.bank_data = pd.concat([new_df, self.bank_data], axis=1)

        return self.bank_data



# bank_data = pd.read_csv('data/bank-additional-full.csv', sep=";")
data = DataProcessing()
print(data.rm_special_char())
test = data.categorical_encode()

# data.categorical_to_numeric()
# numeric_encoder = OneHotEncoder(handle_unknown='ignore')
# transformed = numeric_encoder.fit_transform(bank_data['job']).toarray()
# numeric_encoder.get_feature_names_out(['job'])
