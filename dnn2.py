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
import numpy as np
import joblib
import os


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

# Enable mixed precision
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

# Check for any issues in the columns (optional)
print(data['job_title'].apply(type).unique())  # Ensure all are <class 'str'>
print(data['job_desc'].apply(type).unique())   # Ensure all are <class 'str'>

# Apply the TF-IDF transformation
X_title = vectorizer_title.transform(data['job_title'])
X_desc = vectorizer_desc.transform(data['job_desc'])

# Concatenate the TF-IDF features and the additional numerical features
X = pd.concat([pd.DataFrame(X_title.toarray()), pd.DataFrame(X_desc.toarray()), data[['title_length', 'desc_length']]], axis=1)

# Label encoding for the target variable (if it's categorical)
y = LabelEncoder().fit_transform(data['job_title'])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom data generator to handle memory
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size):
        self.X = X.to_numpy()  # Convert DataFrame to NumPy array
        self.y = y  # `y` is already a NumPy array
        self.batch_size = batch_size

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

# Create data generators for training and testing
batch_size = 8  # Reduced batch size
train_generator = DataGenerator(X_train, y_train, batch_size=batch_size)
test_generator = DataGenerator(X_test, y_test, batch_size=batch_size)

# Define a smaller DNN model architecture
model = Sequential()

# Input Layer (size depends on the number of features after TF-IDF transformation)
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))  # Reduced number of neurons
model.add(Dropout(0.3))

# Hidden Layer 1
model.add(Dense(32, activation='relu'))  # Reduced number of neurons
model.add(Dropout(0.3))

# Hidden Layer 2
model.add(Dense(32, activation='relu'))  # Reduced number of neurons
model.add(Dropout(0.3))

# Output Layer (assuming a classification problem with multiple job categories)
model.add(Dense(len(set(y)), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model using data generators
history = model.fit(train_generator, epochs=20, validation_data=test_generator, 
                    callbacks=[TqdmCallback(), early_stopping])

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator)

print(f'Test Accuracy: {test_accuracy:.2f}')
