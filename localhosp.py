import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23

# Suppress warnings
warnings.filterwarnings('ignore')

# Load your dataset
file_path = r'D:\MAHESH\stress\stress\Worker Stress\LocalModel2_Hospital_worker.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Display the original dataset
print("Original DataFrame:")
print(data.head())

# Drop the columns 'Worker_ID' and 'Timestamp'
columns_to_drop = ['Worker_ID', 'Timestamp']
data = data.drop(columns=columns_to_drop)

# Display the modified dataset
print("\nDataFrame after dropping 'Worker_ID' and 'Timestamp':")
print(data.head())

# Initialize the LabelEncoder
label = LabelEncoder()

# Encode categorical features
data['Shift_Type_label'] = label.fit_transform(data['Shift_Type'])
data['isStressed_label'] = label.fit_transform(data['isStressed'])
data = data.drop(['Shift_Type', 'isStressed'], axis=1)

# Select features for input (x) and target (y)
x = data[['Shift_Type_label', 'PPG_Signal', 'EDA_Signal', 'Blink_Rate', 'RR (Resp. Rate)', 'HRV (Heart Rate Var.)', 'ST (Skin Temp.)', 
          'Motion (Accel.)', 'Voice_Stress', 'Cortisol_Level', 'Stress_Score', 'Stress_Level']]
y = data['isStressed_label']

# Normalize the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Build a custom neural network model for tabular data
model = Sequential()
model.add(Dense(128, input_dim=x_train.shape[1], activation='relu'))  # Input layer with 128 neurons
model.add(Dropout(0.2))  # Dropout layer to avoid overfitting
model.add(BatchNormalization())  # Batch normalization layer
model.add(Dense(64, activation='relu'))  # Hidden layer with 64 neurons
model.add(Dropout(0.2))  # Dropout layer
model.add(BatchNormalization())  # Batch normalization
model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation (binary classification)

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model and save the training history
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model on test data
y_pred = (model.predict(x_test) > 0.5).astype("int32")

# Calculate additional metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = (confusion_matrix(y_test, y_pred).trace() - np.sum(confusion_matrix(y_test, y_pred))) / np.sqrt(
    (np.sum(confusion_matrix(y_test, y_pred, labels=[0,1])[0]) * np.sum(confusion_matrix(y_test, y_pred, labels=[0,1])[1])) 
)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
fnr = fn / (fn + tp)  # False Negative Rate
fpr = fp / (fp + tn)  # False Positive Rate

# Print the performance metrics
print("Performance Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"MCC: {mcc:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"FNR: {fnr:.2f}")
print(f"FPR: {fpr:.2f}")

# Save the metrics to a CSV file
metrics = {
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1 Score': [f1],
    'MCC': [mcc],
    'Specificity': [specificity],
    'FNR': [fnr],
    'FPR': [fpr]
}

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('performance_metrics1.csv', index=False)
print("Metrics saved to performance_metrics1.csv")

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label.classes_)

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)

# Set the x and y ticks labels manually
ax.set_xticklabels(['Stressed', 'Not Stressed'])
ax.set_yticklabels(['Stressed', 'Not Stressed'])

# Set the title and show the plot
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Plot accuracy and loss curves
plt.figure(figsize=(12, 6))

# Accuracy curve
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss curve
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Save the accuracy and loss plots
plt.tight_layout()
plt.savefig('accuracy_loss_curves.png')
plt.show()
