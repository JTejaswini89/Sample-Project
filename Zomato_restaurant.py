import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
tqdm.pandas()

# Load Data
df = pd.read_csv('/content/drive/MyDrive/ZomatoRestuarant.csv')
review_df = pd.read_csv('/content/drive/MyDrive/ Bangalore_Restaurants.csv')
review_df1 = pd.read_csv('/content/drive/MyDrive/ Pune Restaurants.csv')

# Data Cleaning
df.drop(['Collections', 'Links', 'Timings'], axis=1, inplace=True)
df.dropna(inplace=True)
review_df.dropna(inplace=True)
review_df1.dropna(inplace=True)
review_df.drop_duplicates(inplace=True)
review_df1.drop_duplicates(inplace=True)

# Feature Engineering
df.columns = df.columns.str.lower()
df['cost'] = df['cost'].str.replace(",", "").astype('int64')
df['cuisines'] = df['cuisines'].str.replace(' ', '').str.split(',')
review_df.rename(columns={"pricing_for_2": "cost"}, inplace=True)
review_df1.rename(columns={"pricing_for_2": "cost", "known_for1": "Speciality", "known_for2": "Review"}, inplace=True)
review_df['Delivery_Rating'] = review_df['Delivery_Rating'].replace('Like', 4).astype('float64')
review_df['Dining_Rating'] = review_df['Dining_Rating'].replace('Like', 4).astype('float64')
review_df1['Delivery_Rating'] = review_df1['Delivery_Rating'].replace('Like', 4).astype('float64')
review_df1['Dining_Rating'] = review_df1['Dining_Rating'].replace('Like', 4).astype('float64')

# Visualization: Expensive and Least Expensive Restaurants
plt.figure(figsize=(12,6))
sns.barplot(x='cost', y='name', data=df, order=df.sort_values('cost', ascending=False).name[:15])
plt.title('15 Most Expensive Restaurants')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(12,6))
sns.barplot(x='cost', y='name', data=df, order=df.sort_values('cost', ascending=True).name[:15])
plt.title('15 Least Expensive Restaurants')
plt.xticks(rotation=90)
plt.show()

# Cuisine Analysis
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('cuisines')), columns=mlb.classes_, index=df.index))
plt.figure(figsize=(12,10))
sns.barplot(y=df.drop(['name', 'cost'], axis=1).sum().sort_values(ascending=False).index, 
            x=df.drop(['name', 'cost'], axis=1).sum().sort_values(ascending=False))
plt.title('Most Famous Cuisines')
plt.xlabel('Count')
plt.ylabel('Cuisines')
plt.show()

# Ratings Distribution
sns.displot(x=review_df.groupby(['Restaurant_Name'])['Delivery_Rating'].mean(), kde=True)
plt.title('Distribution of Average Restaurant Ratings')
plt.show()

# Merge DataFrames
merge_df = df.merge(review_df, left_on='restaurant', right_on='Restaurant_Name', how='inner') if 'restaurant' in df.columns and 'Restaurant_Name' in review_df.columns else pd.DataFrame()
merge_df1 = df.merge1(review_df, left_on='restaurant', right_on='Restaurant_Name', how='inner') if 'restaurant' in df.columns and 'Restaurant_Name' in review_df.columns else pd.DataFrame()

# Null Value Handling
merge_df.fillna(merge_df.mean(), inplace=True)
merge_df1.fillna(merge_df1.mean(), inplace=True)

# Clustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
scaled_df = df.copy()
scaler = MinMaxScaler()
scaled_df[['cost']] = scaler.fit_transform(scaled_df[['cost']])

wcss = [KMeans(n_clusters=i).fit(scaled_df[['cost']]).inertia_ for i in range(1, 11)]
plt.plot(range(1, 11), wcss, marker='8')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.show()

silhouette = [silhouette_score(scaled_df[['cost']], KMeans(n_clusters=i).fit_predict(scaled_df[['cost']])) for i in range(2, 11)]
plt.plot(range(2, 11), silhouette, marker='8', color='red')
plt.xlabel('Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# KMeans with 5 clusters
km2 = KMeans(n_clusters=5)
scaled_df['labels'] = km2.fit_predict(scaled_df[['cost']])

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca_result = PCA().fit_transform(scaled_df[['cost', 'labels']].dropna().select_dtypes(include=[np.number]))
pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
pca_df['labels'] = km2.labels_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(5):
  ax.scatter(*pca_df.loc[pca_df['labels'] == i, ['PC1', 'PC2']].values.T)
plt.xlabel('PC1')
plt.ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()

# Sentiment Analysis Preparation
merge_df = review_df.reindex(columns=['restaurant_name', 'category', 'cost', 'locality', 'dining_rating', 'dining_review_count', 'delivery_rating', 'delivery_rating_count', 'website', 'address', 'phone_no', 'latitude', 'longitude'])
merge_df1 = review_df1.reindex(columns=['restaurant_name', 'category', 'cost', 'locality', 'dining_rating', 'dining_review_count', 'delivery_rating', 'delivery_rating_count', 'website', 'address', 'phone_no', 'latitude', 'longitude', 'Speciality', 'Review'])

sns.countplot(x=merge_df['cost'])
plt.xticks(rotation=260)
plt.show()
sns.countplot(x=merge_df1['cost'])
plt.xticks(rotation=260)
plt.show()

# Text Preprocessing Function
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import string

def reviews_processing(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Apply Text Preprocessing
review_df['Dining_Review_Count'] = review_df['Dining_Review_Count'].map(reviews_processing)
review_df1['Review'] = review_df1['Review'].apply(reviews_processing) if 'Review' in review_df1.columns else None
