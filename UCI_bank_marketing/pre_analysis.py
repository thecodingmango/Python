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


def scatter_plot(data, x, title, y=None):
    """Returns scatter plot"""

    fig = px.scatter(data, x=x, y=y,
                     title=title)

    return fig.show()


def line_plot(data, x, label, title, y=None):
    """
    Function to plot a histogram given Pandas dataframe
    """
    fig = px.line(data, x=x, y=y,
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

# Histogram of employment variation rate
histogram_plot(banking_data, 'emp.var.rate', bins=5, label='Employment Variation Rate',
               title='Distribution of Employment Variation Rate')

# Histogram of Euro bank interest rate
histogram_plot(banking_data, 'euribor3m', bins=5, label='Interest Rate',
               title='Distribution of Interest Rate')

# Sort months and making it in order
months = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov',
          'dec']
banking_data['month'] = pd.Categorical(banking_data['month'], categories=months, ordered=True)

# Plots consumer price index
line_plot(banking_data.groupby(['month'], as_index=False)['cons.price.idx'].median(),
          x='month',
          y='cons.price.idx',
          title='Trend of Consumer Price Index',
          label='Month')

# Plots consumer confidence index
line_plot(banking_data.groupby(['month'], as_index=False)['cons.conf.idx'].median(),
          x='month',
          y='cons.conf.idx',
          title='Trend of Consumer Confidence Index',
          label='Month')

# Plots consumer confidence index
line_plot(banking_data.groupby(['month'], as_index=False)['nr.employed'].median(),
          x='month',
          y='nr.employed',
          title='Trend of Number of Employed',
          label='# of Employed')



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









