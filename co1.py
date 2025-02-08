import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pywt
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Load dataset
df = pd.read_csv(r"D:\MAHESH\stress\stress\Worker Stress\stress2\stress_detection.csv")

# Drop unnecessary columns
df = df.drop(columns=['participant_id', 'day'])

# Define x and y labels
x_columns = ['PSS_score', 'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism', 
             'sleep_time', 'wake_time', 'sleep_duration', 'call_duration', 'num_calls', 'num_sms', 'screen_on_time',
             'skin_conductance', 'accelerometer', 'mobility_radius', 'mobility_distance']
y_column = 'PSQI_score'

X = df[x_columns]
y = df[y_column]

# Bandpass Filter Function
def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=200, order=5):
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0 or high >= 1:
        raise ValueError("Adjusted cutoff frequencies must be between 0 and 1")
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Apply filtering to physiological signals
signal_columns = ['skin_conductance', 'accelerometer']
for col in signal_columns:
    df[col] = bandpass_filter(df[col])

# Normalization
scaler = MinMaxScaler()
df[x_columns] = scaler.fit_transform(df[x_columns])

# Encode labels
le = LabelEncoder()
df[y_column] = le.fit_transform(df[y_column])

# **Check Unique Class Count**
num_classes = df[y_column].nunique()
print(f"Number of unique classes in {y_column}: {num_classes}")

# Feature Extraction using Wavelet Transform
def wavelet_transform(data, wavelet='db4', level=3):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return np.concatenate([np.ravel(c) for c in coeffs])

wavelet_features = df[x_columns].apply(lambda x: wavelet_transform(x), axis=1, result_type='expand')

# Feature Selection using RFE
rfe = RFE(RandomForestClassifier(n_estimators=100), n_features_to_select=10)
X_selected = rfe.fit_transform(wavelet_features, df[y_column])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_selected, df[y_column], test_size=0.2, random_state=84)

# **Use Dynamic num_classes**
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# Define Optimized Neural Network Model
model = Sequential([
    Dense(256, activation=tf.keras.activations.swish, kernel_regularizer=l2(0.01), input_dim=X_train.shape[1]),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(128, activation=tf.keras.activations.swish, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(64, activation=tf.keras.activations.swish, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(num_classes, activation='softmax')  # Adjusted dynamically
])

# **Learning Rate Scheduler & Early Stopping**
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Compile the model with improved learning rate
model.compile(optimizer=Adam(learning_rate=0.005),  # Increased initial learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with smaller batch size
history = model.fit(X_train, y_train_cat,
                    epochs=100,
                    batch_size=64,  # Reduced batch size
                    validation_data=(X_test, y_test_cat),
                    callbacks=[reduce_lr, early_stop])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model
model.save(r"D:\MAHESH\stress\stress\Worker Stress\stress2\stress_model_optimized.h5")
print("\nOptimized Model saved successfully!")
