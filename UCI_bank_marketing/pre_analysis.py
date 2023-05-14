"""
Main File for exploring the characteristics of the data
"""

# Importing necessary libraries
import pandas as pd
import numpy as np
import plotly.express as px

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

"""
# Exploratory Data Analysis
# Section with various histogram plots
"""


# Given data, x, y (optional),# of bins, label, title, returns histogram
def histogram_plot(data, x, bins, label, title, y=None):
    """
    Function to plot a histogram given Pandas dataframe
    """
    fig = px.histogram(data, x=x, y=y, nbins=bins,
                       title=title,
                       labels={
                           x: label
                       })
    fig.update_layout(bargap=0.2)
    return fig.show()


# Plotting histogram plots
histogram_plot(banking_data, 'age', bins=30, label='Age of Client',
               title='Distribution of Age for Client Contacted by Bank')

# Histogram of Duration
histogram_plot(banking_data, 'duration', bins=100, label='Duration of call',
               title='Distribution of Duration of Previous Call')

# Histogram of Number of contact during campaign
histogram_plot(banking_data, 'campaign', bins=100, label='Number of Contact for the Duration of Campaign',
               title='Distribution of Number of Telephone Contacts')

# Histogram of Number of days from the previous contact
histogram_plot(banking_data, 'pdays', bins=100, label='Number of Days from the Previous Contact',
               title='Distribution of Number of Telephone Contacts')


# Histogram of number of contact before the campaign
histogram_plot(banking_data, 'previous', bins=10, label='Number of contact before the campaign',
               title='Distribution of Number of Telephone Contacts before the Campaign')






# Histogram of Job
fig = px.histogram(banking_data, x='job', nbins=10, color='job',
                   title='Distribution of Job for Client Contacted by Bank',
                   category_orders=dict(job=['housemaid', 'services', 'admin.', 'blue-collar', 'technician', 'retired',
                                             'management', 'unemployed', 'self-employed', 'unknown', 'entrepreneur',
                                             'student']),
                   labels={
                       'job': 'Job of the Person Contacted'
                   }
                   )
fig.update_layout(bargap=0.2)
fig.show()









