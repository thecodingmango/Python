"""
This file will help to process the data before using it for logistic regression
"""
import pandas as pd
from pandas.api.types import is_object_dtype
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


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

    def month_encode(self):
        """
        Encodes the month using numeric in strings
        """

        months = {'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                  'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

        self.bank_data['month'] = self.bank_data['month'].map(months)

        start_year = 2008
        curr_month = 5
        year_list = []

        for _, row in self.bank_data.iterrows():

            if int(row['month']) == curr_month:

                year_list += [start_year]

            elif int(row['month']) < curr_month:

                start_year += 1
                curr_month = int(row['month'])
                year_list += [start_year]

            if int(row['month']) > curr_month:
                year_list += [start_year]
                curr_month += 1

        self.bank_data['year'] = year_list

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
                elif self.bank_data[column].nunique() > 2 and column != 'month':

                    oe_encoder = OneHotEncoder(handle_unknown='ignore')

                    col_name = [column + '_' + name for name in sorted(self.bank_data[column].unique())]
                    new_df = pd.DataFrame(oe_encoder.fit_transform(self.bank_data[[column]]).toarray(),
                                          columns=col_name)
                    new_df = new_df.drop(new_df.columns[0], axis=1)

                    self.bank_data = self.bank_data.drop(column, axis=1)

                    self.bank_data = pd.concat([new_df, self.bank_data], axis=1)

        return self.bank_data

    def min_max_scaler(self):

        for column in self.bank_data.iloc[:, 1:]:

            min_val = min(self.bank_data[column])

            max_val = max(self.bank_data[column])

            scaled = (self.bank_data[column] - min_val) / (max_val - min_val)

            self.bank_data[column] = scaled

        return self.bank_data

    def preprocessing(self):

        # Removes special characters
        DataProcessing.rm_special_char(self)

        # Encode categorical characters
        DataProcessing.categorical_encode(self)

        # Encode dates
        DataProcessing.month_encode(self)

        # Drop Duration column
        self.bank_data = self.bank_data.drop("duration", axis=1)

        DataProcessing.min_max_scaler(self)

        return pd.DataFrame(self.bank_data)

# bank_data = pd.read_csv('data/bank-additional-full.csv', sep=";")
# data = DataProcessing()
# print(data.rm_special_char())
# test = data.categorical_encode()
#test2 = pd.read_csv('data/bank-additional-full.csv', sep=";")
#test = DataProcessing()
#test = test.preprocessing()
# print(data.normalization('age'))
# print(data.standardization('age'))
