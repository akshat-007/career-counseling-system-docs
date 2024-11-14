# Import necessary libraries
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.mixed_precision import set_global_policy

from tqdm.keras import TqdmCallback

import joblib

# List physical devices
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs available")

# Set mixed precision policy
policy = 'mixed_float16'
set_global_policy(policy)

# Load preprocessed data
file_path = '/Users/akshu/career_counseling_system/Datasets/preprocessed_data.csv'
data = pd.read_csv(file_path)

# Load saved TF-IDF vectorizers
vectorizer_title = joblib.load('vectorizer_title.pkl')
vectorizer_desc = joblib.load('vectorizer_desc.pkl')

# Ensure job_title and job_desc columns contain strings
data['job_title'] = data['job_title'].astype(str)
data['job_desc'] = data['job_desc'].astype(str)

# Check for any issues in the columns 
print(data['job_title'].apply(type).unique())  # Ensure all are <class 'str'>
print(data['job_desc'].apply(type).unique())   # Ensure all are <class 'str'>

# Now, apply the TF-IDF transformation
X_title = vectorizer_title.transform(data['job_title'])
X_desc = vectorizer_desc.transform(data['job_desc'])

# Concatenate the TF-IDF features and the additional numerical features
X = pd.concat([pd.DataFrame(X_title.toarray()), pd.DataFrame(X_desc.toarray()), data[['title_length', 'desc_length']]], axis=1)

# target variable is 'job_title' or 'job_type' (replace this with your actual target column)
# Label encoding for the target variable (if it's categorical)
y = LabelEncoder().fit_transform(data['job_title'])  

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the DNN model architecture
model = Sequential()

# Input Layer (size depends on the number of features after TF-IDF transformation)
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))  # Using Input layer to specify the input shape
model.add(Dense(256, activation='relu'))  # Now you can omit input_dim
model.add(Dropout(0.3))  # Dropout to prevent overfitting

# Hidden Layer 1
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

# Hidden Layer 2
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

# Output Layer (assuming a classification problem with multiple job categories)
model.add(Dense(len(set(y)), activation='softmax'))  # Use 'softmax' for multi-class classification

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), 
                    callbacks=[TqdmCallback(), early_stopping])
# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f'Test Accuracy: {test_accuracy:.2f}')