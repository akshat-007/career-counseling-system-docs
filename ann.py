import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from scipy.sparse import hstack

# Load saved TF-IDF vectorizers
vectorizer_title = joblib.load('vectorizer_title.pkl')
vectorizer_desc = joblib.load('vectorizer_desc.pkl')


# Load your preprocessed data
df = pd.read_csv('/Users/akshu/career_counseling_system/Datasets/preprocessed_data.csv')
data = pd.read_csv(file_path)

# Transform job titles and descriptions into TF-IDF vectors
X_title = vectorizer_title.transform(data['job_title'])
X_desc = vectorizer_desc.transform(data['job_desc'])

# Combine the TF-IDF vectors into a single feature matrix
X = hstack([X_title, X_desc])

# Encode the job titles as integers
y = df['job_title']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the ANN model
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))  # Input layer
model.add(Dropout(0.3))  # Dropout layer for regularization
model.add(Dense(128, activation='relu'))  # Hidden layer
model.add(Dropout(0.3))  # Dropout layer
model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
