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
def histogram_plot(data, x, bins, label, title, y=None, category=None, color=None, pattern=None, marginal=None):
    """
    Function to plot a histogram given Pandas dataframe
    """
    fig = px.histogram(data,
                       x=x,
                       y=y,
                       nbins=bins,
                       color=color,
                       category_orders=dict(job=category),
                       title=title,
                       labels={
                           x: label
                       },
                       pattern_shape=pattern,
                       marginal=marginal)
    fig.update_layout(bargap=0.2)
    return fig.show()


def scatter_plot(data, x, title, y=None, color=None, size=None, x_label=None, y_label=None):
    """Returns scatter plot"""

    fig = px.scatter(
        data,
        x=x,
        y=y,
        color=color,
        size=size,
        title=title
                     )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label
    )

    return fig.show()


def line_plot(data, x, label, title, y=None, color=None):
    """
    Function to plot a histogram given Pandas dataframe
    """
    fig = px.line(data,
                  x=x,
                  y=y,
                  title=title,
                  color=color,
                  labels={
                      x: label
                  })
    fig.update_layout(bargap=0.2)
    return fig.show()


# Plotting histogram plots
histogram_plot(banking_data, x='age', bins=30, label='Age of Client', color='y',
               title='Distribution of Age for Client Contacted by Bank', marginal='violin')

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
histogram_plot(banking_data, x='euribor3m', bins=100, label='Interest Rate', color='y',
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

# Plots number of employed at the bank
line_plot(banking_data.groupby(['month'], as_index=False)['nr.employed'].median(),
          x='month',
          y='nr.employed',
          title='Trend of Number of Employed',
          label='# of Employed')

"""
This Section is for the plotting the bar chart for the categorical variables
"""


# Histogram of Job
histogram_plot(banking_data, x='job', bins=10, color='y',
               title='Distribution of Job for Client Contacted by Bank',
               category=['housemaid', 'services', 'admin.', 'blue-collar', 'technician', 'retired',
                         'management', 'unemployed', 'self-employed', 'unknown', 'entrepreneur',
                         'student'],
               label='Job of the Person Contacted')

# Histogram of marital
histogram_plot(banking_data, x='marital', bins=10, color='marital',
               title='Distribution of Marital Status for Client Contacted by Bank',
               category=['divorced', 'married', 'single', 'unknown'],
               label='Marital Status')

# Histogram of Education
histogram_plot(banking_data, x='education', bins=10, color='y',
               title='Distribution of Education for Client Contacted by Bank',
               category=['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                         'illiterate', 'professional.course', 'university.degree', 'unknown'],
               label='Education Level')

# Histogram of Default
histogram_plot(banking_data, x='default', bins=10, color='default',
               title='Distribution of Default Status for Client Contacted by Bank',
               category=['no', 'yes', 'unknown'],
               label='Previously Defaulted')

# Histogram of housing loan
histogram_plot(banking_data, x='housing', bins=10, color='housing',
               title='Distribution of Housing Loan Type for Client Contacted by Bank',
               category=['no', 'yes', 'unknown'],
               label='Have Housing Loan')

# Histogram of loan type
histogram_plot(banking_data, x='loan', bins=10, color='loan',
               title='Distribution of Loan Type for Client Contacted by Bank',
               category=['no', 'yes', 'unknown'],
               label='Loan Type')

# Histogram of Outcome
histogram_plot(banking_data, x='y', bins=10, color='y',
               title='Outcome of Term Deposit for Client Contacted by Bank',
               category=['yes', 'no'],
               label='Term Deposit')

scatter_plot(banking_data, x=banking_data.groupby(['month'], as_index=False)['cons.price.idx'].median()['cons.price.idx'],
             y=banking_data.groupby(['month'], as_index=False)['euribor3m'].median()['euribor3m'],
             color=banking_data.groupby(['month'], as_index=False)['euribor3m'].median()['month'],
             title='Scatter Plot Comparing CPI and Interest Rate by Month', x_label='Consumer Price Index',
             y_label='Interest Rate')
