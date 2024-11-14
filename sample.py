# Step 1: Import necessary libraries
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback
from sklearn.utils import class_weight
import joblib
import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt

# Step 2: List physical devices and set mixed precision
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs available")

# Step 3: Load preprocessed data
file_path = '/Users/akshu/career_counseling_system/Datasets/preprocessed_data.csv'  
data = pd.read_csv(file_path)

# Step 4: Load saved TF-IDF vectorizers
vectorizer_title = joblib.load('vectorizer_title.pkl')  # Adjust the path if needed
vectorizer_desc = joblib.load('vectorizer_desc.pkl')  # Adjust the path if needed

# Step 5: Ensure job_title and job_desc columns contain strings
data['job_title'] = data['job_title'].astype(str)
data['job_desc'] = data['job_desc'].astype(str)

# Step 6: Sample the data
sample_df = data.sample(n=1000, random_state=42)  # Sample 1000 rows

# Step 7: Apply the TF-IDF transformation
X_title = vectorizer_title.transform(sample_df['job_title'])
X_desc = vectorizer_desc.transform(sample_df['job_desc'])

# Convert the TF-IDF matrices to DataFrames and reset index
X_title_df = pd.DataFrame(X_title.toarray()).reset_index(drop=True)
X_desc_df = pd.DataFrame(X_desc.toarray()).reset_index(drop=True)

# Step 8: Scale numerical features (title_length and desc_length)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(sample_df[['title_length', 'desc_length']])

# Step 9: Concatenate the TF-IDF features and scaled numerical features
X = pd.concat([X_title_df, X_desc_df, pd.DataFrame(scaled_features, columns=['title_length', 'desc_length'])], axis=1)

# Convert all column names to strings
X.columns = X.columns.astype(str)  # Ensure all column names are of type string


# Step 10: Prepare the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(sample_df['job_title'])  # Ensure y is also from sample_df

# Check class distribution
class_counts = Counter(y)
print(f'Class distribution before oversampling: {class_counts}')

# Step 11: Handle class imbalance using oversampling
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Check class distribution after oversampling
class_counts_resampled = Counter(y_resampled)
print(f'Class distribution after oversampling: {class_counts_resampled}')

# Step 12: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 13: Compute class weights (optional if oversampling doesn't help)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Step 14: Define the DNN model architecture with more layers and batch normalization
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))  # Input layer

# Added hidden layers with Batch Normalization
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))

# Output layer
model.add(Dense(len(set(y)), activation='softmax'))

# Step 15: Compile the model with Adam optimizer and a higher learning rate
from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001)  # Increased learning rate

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Step 16: Define EarlyStopping with a higher patience
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Step 17: Train the model for more epochs and include progress bar
history = model.fit(
    X_train, y_train, 
    epochs=100, batch_size=32,  # Increased number of epochs
    validation_data=(X_test, y_test), 
    class_weight=class_weights_dict,  # Use class weights to handle imbalance (optional with oversampling)
    callbacks=[early_stopping, TqdmCallback()]
)

# Step 18: Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Step 19: Save the trained model
model.save('trained_dnn_model')

 # Save in the recommended Keras format

# Step 20: Save the label encoder and scaler for future use
joblib.dump(label_encoder, 'label_encoder_v2.pkl')  # Save the label encoder
joblib.dump(scaler, 'scaler_v2.pkl')  # Save the scaler for numerical feature scaling

# Step 21: Plot training history for accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
