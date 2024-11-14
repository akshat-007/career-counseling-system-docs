import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.mixed_precision import set_global_policy
from tqdm.keras import TqdmCallback
import numpy as np
import joblib
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

# Enable mixed precision
set_global_policy('mixed_float16')

# Load preprocessed data
data = pd.read_csv('/Users/akshu/career_counseling_system/Datasets/preprocessed_data.csv')
vectorizer_title = joblib.load('vectorizer_title.pkl')
vectorizer_desc = joblib.load('vectorizer_desc.pkl')

# Ensure job_title and job_desc columns contain strings
data['job_title'] = data['job_title'].astype(str)
data['job_desc'] = data['job_desc'].astype(str)

# Apply the TF-IDF transformation
X_title = vectorizer_title.transform(data['job_title'])
X_desc = vectorizer_desc.transform(data['job_desc'])
X = pd.concat([pd.DataFrame(X_title.toarray()), pd.DataFrame(X_desc.toarray()), data[['title_length', 'desc_length']]], axis=1)
y = LabelEncoder().fit_transform(data['job_title'])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom data generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size):
        self.X = X.to_numpy()
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

# Create data generators
batch_size = 4  # Further reduced
train_generator = DataGenerator(X_train, y_train, batch_size=batch_size)
test_generator = DataGenerator(X_test, y_test, batch_size=batch_size)

# Define model architecture
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(set(y)), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(train_generator, epochs=20, validation_data=test_generator, 
                    callbacks=[TqdmCallback(), early_stopping, checkpoint], verbose=1)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy:.2f}')
