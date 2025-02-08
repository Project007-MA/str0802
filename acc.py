import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
import pywt  # PyWavelets for Wavelet Transform
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Suppress warnings
warnings.filterwarnings('ignore')

# Load your dataset
file_path = r'D:\MAHESH\stress\stress\Worker Stress\LocalModel1_IT_Worker.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Drop the columns 'Worker_ID' and 'Timestamp' if they exist
columns_to_drop = ['Worker_ID', 'Timestamp']
existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]

# Drop the existing columns
data = data.drop(columns=existing_columns_to_drop)

# Initialize the LabelEncoder
label = LabelEncoder()

# Encode categorical features
data['Shift_Type_label'] = label.fit_transform(data['Shift_Type'])
data['isStressed_label'] = label.fit_transform(data['isStressed'])
data = data.drop(['Shift_Type', 'isStressed'], axis=1)

# Select features for input (x) and target (y)
x = data[['PPG_Signal', 'EDA_Signal','Stress_Level']]  # Adjust based on your signals
y = data['isStressed_label']

# Wavelet Transform Feature Extraction
def wavelet_features(signal, wavelet='db1', level=3):
    """
    Extract features using Discrete Wavelet Transform (DWT).
    :param signal: Input signal (1D array)
    :param wavelet: Wavelet type (e.g., 'db1', 'haar')
    :param level: Decomposition level
    :return: Extracted coefficients (approximation and details)
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = []
    for coeff in coeffs:
        features.extend([np.mean(coeff), np.std(coeff)])  # Mean and Std as features
    return features

# Apply Wavelet Transform to each row in the dataset
wavelet_features_list = []
for _, row in x.iterrows():
    row_features = []
    for column in x.columns:
        signal = row[column]  # Process each signal
        row_features.extend(wavelet_features([signal], wavelet='db1', level=3))  # Ensure input is 1D array
    wavelet_features_list.append(row_features)

# Convert wavelet features to DataFrame
wavelet_features_df = pd.DataFrame(wavelet_features_list)
wavelet_features_df.columns = [f'feature_{i}' for i in range(wavelet_features_df.shape[1])]

# Ensure consistency in dimensions
X_features = wavelet_features_df
y_labels = y.reset_index(drop=True)

# Feature Selection with RFE
# Using RandomForestClassifier as the estimator
estimator = RandomForestClassifier(random_state=42)
rfe = RFE(estimator, n_features_to_select=5)  # Select top 5 features (adjust as needed)
X_selected = rfe.fit_transform(X_features, y_labels)

# Display selected features
selected_feature_names = wavelet_features_df.columns[rfe.support_]
print("\nSelected Features via RFE:")
print(selected_feature_names)

# Combine selected features into a DataFrame
selected_features_df = pd.DataFrame(X_selected, columns=selected_feature_names)

# Combine selected features with target labels
final_data = pd.concat([selected_features_df, y_labels], axis=1)

# Display final data
print("\nFinal Data with Selected Features:")
print(final_data.head())

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_labels, test_size=0.2, random_state=42)

# Convert labels to categorical for TensorFlow
y_train_cat = to_categorical(y_train, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)

# Check the shape of X_train
print(f"X_train shape before reshaping: {X_train.shape}")

# Reshape the data to be compatible with a dense network (flattening features)
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1])  # Flattening to (batch_size, num_features)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1])

# Now create the model
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train_reshaped.shape[1]),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Adjust for binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train_reshaped, y_train_cat, 
                    epochs=20, 
                    batch_size=32, 
                    validation_data=(X_test_reshaped, y_test_cat))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test_cat)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# Optional: Save the trained model
model.save(r'D:\MAHESH\stress\stress\Worker Stress\ResNet101_CNN_Model.h5')
print("\nModel saved successfully!")
