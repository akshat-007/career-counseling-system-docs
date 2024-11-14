# Import necessary libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
# Load the dataset
file_path = '/Users/akshu/career_counseling_system/Datasets/clean_data.parquet'
data = pd.read_parquet(file_path)
# Drop missing values
data.dropna(subset=['job_title', 'job_desc'], inplace=True)
 #Remove duplicates
data.drop_duplicates(subset=['job_title', 'job_desc'], inplace=True)
# Text normalization
data['job_title'] = data['job_title'].str.lower().apply(lambda x: re.sub(r'[^a-z\s]', '', x))
data['job_desc'] = data['job_desc'].str.lower().apply(lambda x: re.sub(r'[^a-z\s]', '', x))
# Tokenization
data['job_title'] = data['job_title'].apply(lambda x: x.split())
data['job_desc'] = data['job_desc'].apply(lambda x: x.split())
# Stop word removal
stop_words = set(stopwords.words('english'))
data['job_title'] = data['job_title'].apply(lambda x: [word for word in x if word not in stop_words])
data['job_desc'] = data['job_desc'].apply(lambda x: [word for word in x if word not in stop_words])
# Stemming
stemmer = PorterStemmer()
data['job_title'] = data['job_title'].apply(lambda x: [stemmer.stem(word) for word in x])
data['job_desc'] = data['job_desc'].apply(lambda x: [stemmer.stem(word) for word in x])
# Feature engineering
data['title_length'] = data['job_title'].apply(len)
data['desc_length'] = data['job_desc'].apply(len)
# Vectorization
data['job_title'] = data['job_title'].apply(lambda x: ' '.join(x))
data['job_desc'] = data['job_desc'].apply(lambda x: ' '.join(x))
vectorizer = TfidfVectorizer()
X_title = vectorizer.fit_transform(data['job_title'])
X_desc = vectorizer.fit_transform(data['job_desc'])
# Save the preprocessed data
data.to_csv('/Users/akshu/career_counseling_system/Datasets/preprocessed_data.csv', index=False)