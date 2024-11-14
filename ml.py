import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load the preprocessed data
df = pd.read_csv('/Users/akshu/career_counseling_system/Datasets/preprocessed_data.csv')

# Check for missing values in 'job_desc' and 'job_title' columns
missing_desc = df['job_desc'].isnull().sum()
missing_title = df['job_title'].isnull().sum()

print(f"Found {missing_desc} missing values in 'job_desc' column.")
print(f"Found {missing_title} missing values in 'job_title' column.")

# Option 1: Drop rows with missing 'job_desc' or 'job_title'
df = df.dropna(subset=['job_desc', 'job_title'])

# Load the saved TF-IDF vectorizers
vectorizer_title = joblib.load('vectorizer_title.pkl')
vectorizer_desc = joblib.load('vectorizer_desc.pkl')

# Transform the job descriptions and job titles into TF-IDF vectors
tfidf_matrix_desc = vectorizer_desc.transform(df['job_desc'])
tfidf_matrix_title = vectorizer_title.transform(df['job_title'])

# Compute cosine similarity for job descriptions
cosine_sim_desc = cosine_similarity(tfidf_matrix_desc, tfidf_matrix_desc)

# Define the recommendation function
def get_recommendations(job_title, cosine_sim=cosine_sim_desc, data=df):
    # Find the index of the job that matches the title
    try:
        idx = data[data['job_title'] == job_title].index[0]
    except IndexError:
        return "Job title not found."

    # Get the similarity scores for this job with all others
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort jobs based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top 10 most similar jobs (excluding the first one)
    sim_scores = sim_scores[1:11]  # Top 10

    # Get the job indices
    job_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar jobs
    return data['job_title'].iloc[job_indices]

# Example usage (optional)
job_recommendations = get_recommendations('Data Scientist')
print(job_recommendations)

# Save the model (TF-IDF vectorizers and cosine similarity matrix)
model_data = {
    'vectorizer_desc': vectorizer_desc,
    'vectorizer_title': vectorizer_title,
    'cosine_sim_desc': cosine_sim_desc
}

# Path to save the model
save_path = '/Users/akshu/career_counseling_system/model_code/your_model.pkl'

# Save the model to a pickle file
with open(save_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"Model saved successfully at {save_path}")
