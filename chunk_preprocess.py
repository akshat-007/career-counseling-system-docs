# Import necessary libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import joblib

nltk.download('stopwords')

# Function to preprocess each chunk
def preprocess_chunk(chunk):
    # Drop missing values
    chunk.dropna(subset=['job_title', 'job_desc'], inplace=True)

    # Remove duplicates
    chunk.drop_duplicates(subset=['job_title', 'job_desc'], inplace=True)

    # Text normalization
    chunk['job_title'] = chunk['job_title'].str.lower().apply(lambda x: re.sub(r'[^a-z\s]', '', x))
    chunk['job_desc'] = chunk['job_desc'].str.lower().apply(lambda x: re.sub(r'[^a-z\s]', '', x))

    # Tokenization
    chunk['job_title'] = chunk['job_title'].apply(lambda x: x.split())
    chunk['job_desc'] = chunk['job_desc'].apply(lambda x: x.split())

    # Stop word removal
    stop_words = set(stopwords.words('english'))
    chunk['job_title'] = chunk['job_title'].apply(lambda x: [word for word in x if word not in stop_words])
    chunk['job_desc'] = chunk['job_desc'].apply(lambda x: [word for word in x if word not in stop_words])

    # Stemming
    stemmer = PorterStemmer()
    chunk['job_title'] = chunk['job_title'].apply(lambda x: [stemmer.stem(word) for word in x])
    chunk['job_desc'] = chunk['job_desc'].apply(lambda x: [stemmer.stem(word) for word in x])

    # Feature engineering: add title and description length
    chunk['title_length'] = chunk['job_title'].apply(len)
    chunk['desc_length'] = chunk['job_desc'].apply(len)

    # Convert back to string for TF-IDF
    chunk['job_title'] = chunk['job_title'].apply(lambda x: ' '.join(x))
    chunk['job_desc'] = chunk['job_desc'].apply(lambda x: ' '.join(x))

    return chunk

# File path for large dataset
file_path = '/Users/akshu/career_counseling_system/Datasets/clean_data.parquet'  # Update with your file path

# Output file for preprocessed data
output_file = '/Users/akshu/career_counseling_system/Datasets/preprocessed_data.csv'  # Update with your output file path

# Set chunk size to process data in manageable parts
chunksize = 5000  # This processes 5000 rows at a time; you can adjust it

# Initialize vectorizers outside the loop
vectorizer_title = TfidfVectorizer()
vectorizer_desc = TfidfVectorizer()

# Read parquet file in chunks manually since parquet doesn't support chunking
data = pd.read_parquet(file_path)
num_chunks = len(data) // chunksize + 1

# Loop over manually created chunks
for i in range(num_chunks):
    chunk = data.iloc[i*chunksize:(i+1)*chunksize]
    
    # Preprocess the current chunk
    chunk = preprocess_chunk(chunk)
    
    # Vectorization using TF-IDF
    X_title = vectorizer_title.fit_transform(chunk['job_title'])
    X_desc = vectorizer_desc.fit_transform(chunk['job_desc'])
    
    # Save the preprocessed chunk to CSV (append mode)
    chunk.to_csv(output_file, mode='a', header=(i == 0), index=False)

# Save TF-IDF vectorizers for future use
joblib.dump(vectorizer_title, 'vectorizer_title.pkl')
joblib.dump(vectorizer_desc, 'vectorizer_desc.pkl')

print("Batch processing completed and data saved.")
