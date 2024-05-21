import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
OR_df = pd.read_csv("/content/drive/MyDrive/Online_Retail.csv") # change the file name as per your data source
# Data preprocessing
OR_df['InvoiceDate'] = pd.to_datetime(OR_df['InvoiceDate']) # convert to datetime format
OR_df = OR_df[OR_df['Quantity'] > 0] # remove negative quantities
OR_df = OR_df[OR_df['CustomerID'].notnull()] # remove missing customer ids
OR_df['TotalAmount'] = OR_df['Quantity'] * OR_df['UnitPrice'] # calculate total amount for each transaction

# RFM analysis
# Define a snapshot date as the day after the last transaction in the data
snapshot_date = OR_df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Aggregate data by customer id and calculate recency, frequency and monetary value
rfm_df = OR_df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # recency: number of days since last purchase
    'InvoiceNo': 'count', # frequency: number of transactions
    'TotalAmount': 'sum' # monetary value: total amount spent
})

# Rename the columns
rfm_df.rename(columns={'InvoiceDate': 'Recency',
                    'InvoiceNo': 'Frequency',
                    'TotalAmount': 'Monetary'}, inplace=True)

# RFM segmentation
# Define quartiles for each metric
r_quartiles = pd.qcut(rfm_df['Recency'], 4, labels=[4, 3, 2, 1]) # higher recency means lower value
f_quartiles = pd.qcut(rfm_df['Frequency'], 4, labels=[1, 2, 3, 4]) # higher frequency means higher value
m_quartiles = pd.qcut(rfm_df['Monetary'], 4, labels=[1, 2, 3, 4]) # higher monetary value means higher value

# Assign quartiles to each customer
rfm_df['R'] = r_quartiles
rfm_df['F'] = f_quartiles
rfm_df['M'] = m_quartiles

# Concatenate the quartiles to get the RFM score for each customer
rfm_df['RFM_Score'] = rfm_df[['R', 'F', 'M']].sum(axis=1)

# Define RFM segments based on RFM score ranges
def rfm_segment(score):
    if score >= 9:
        return "Major customers"
    elif score >= 8:
        return "Loyal Customers"
    elif score >= 7:
        return "Potential Loyalists"
    elif score >= 6:
        return "Promising"
    elif score >= 5:
        return "Needs Atention"
    elif score >= 4:
        return "About To Leave"
    elif score >= 3:
        return "At Risk"
    elif score >= 2:
        return "Can't Lose Them"
    else:
        return "Lost"

# Assign segments to each customer
rfm_df['RFM_Segment'] = rfm_df['RFM_Score'].apply(rfm_segment)

# Display the RFM table
print(rfm_df.head())

# Visualize the RFM segments
sns.countplot(x='RFM_Segment', data=rfm_df, order=rfm_df['RFM_Segment'].value_counts().index, palette='Set1')
plt.xticks(rotation=90)
plt.plot()
plt.show()