# Importing necessary libraries
import numpy as np
from data_processing import DataProcessing
from Visualizations import histogram_plot, line_plot, scatter_plot

# Future Use
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Read the data from the csv
banking_data = DataProcessing()
banking_data = banking_data.preprocessing()
banking_data = banking_data.loc[:, [
                                       'y', 'year', 'month', 'poutcome_nonexistent', 'poutcome_success',
                                       'day_of_week_mon',
                                       'day_of_week_thu', 'day_of_week_tue', 'day_of_week_wed', 'loan_unknown',
                                       'loan_yes', 'housing_unknown', 'housing_yes', 'default_unknown',
                                       'default_yes', 'education_basic_6y', 'education_basic_9y',
                                       'education_high_school', 'education_illiterate',
                                       'education_professional_course', 'education_university_degree',
                                       'education_unknown', 'marital_married', 'marital_single',
                                       'marital_unknown', 'job_blue_collar', 'job_entrepreneur',
                                       'job_housemaid', 'job_management', 'job_retired', 'job_self_employed',
                                       'job_services', 'job_student', 'job_technician', 'job_unemployed',
                                       'job_unknown', 'age', 'contact', 'campaign', 'pdays',
                                       'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                                       'euribor3m', 'nr.employed']]


"""
PCA Data Analysis
"""

# Generate the variance covariance matrix
var_mat = banking_data.iloc[:, 1:].cov()

# Decomposing the variance covariance matrix into eigenvalues, and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(var_mat)

# Calculating the proportion of variance explained by each eigenvalue
prop_var = eigenvalues / sum(eigenvalues)

# Calculating total variance explained up to i-th eigenvalue
total_prop_var = list()

for i in range(1, 46):
    total_prop_var += [sum(prop_var[0:i])]

# Plotting the total variance against number of eigenvalue
scatter_plot(None, range(1, 46), "Comparison of # of Principal Component and Total Variance Explained",
             y=total_prop_var, x_label="# of Principal Component", y_label="Total Variance Explained")

# Choosing first 25 Principal Component
bank_data_new = banking_data.iloc[:, 1:46].dot(eigenvectors[:, 0:24])
bank_data_new['y'] = banking_data['y']


"""
Future Use
"""
# # Split the data into train and test set
# x = banking_data.loc[:, banking_data.columns != 'y']
# y = banking_data['y']
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
#
# column_list = ['age', 'campaign', 'pdays', 'emp.var.rate', 'cons.price.idx', 'euribor3m', 'nr.employed']
# x_train = min_max_scaler(x_train, column_list)
