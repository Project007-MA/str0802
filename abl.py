import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pywt  # PyWavelets for Wavelet Transform
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Suppress warnings
warnings.filterwarnings('ignore')

# Load your dataset
file_path = r'D:\MAHESH\stress\stress\Worker Stress\LocalModel1_IT_Worker.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Drop the columns 'Worker_ID' and 'Timestamp'
columns_to_drop = ['Worker_ID', 'Timestamp']
data = data.drop(columns=columns_to_drop)

# Initialize the LabelEncoder
label = LabelEncoder()

# Encode categorical features
data['Shift_Type_label'] = label.fit_transform(data['Shift_Type'])
data['isStressed_label'] = label.fit_transform(data['isStressed'])
data = data.drop(['Shift_Type', 'isStressed'], axis=1)

# Select features for input (x) and target (y)
x = data[['PPG_Signal', 'EDA_Signal']]  # Adjust based on your signals
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

# Optional: Save the final data to a CSV file
output_file_path = r'D:\MAHESH\stress\stress\Worker Stress\Processed_Final_Data.csv'
final_data.to_csv(output_file_path, index=False)
print(f"\nFinal data saved to: {output_file_path}")
