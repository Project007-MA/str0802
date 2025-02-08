import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23


# Classifier names
classifiers = ['CNN-LSTM', 'LSTM, SVM, and CNN', 'MSD', 'StressNet', 'ResTFTNet (Proposed)']

# Accuracy values (in percentages)
accuracy_values = [83.88, 87, 96, 97.8, 99]

# Set width for the bars
bar_width = 0.5
index = np.arange(len(classifiers))

# Create the plot
plt.figure(figsize=(10, 6))

# Plot horizontal bars for classifier accuracies
bars = plt.barh(index, accuracy_values, bar_width, color='skyblue', label='Accuracy')

# Add labels, title, and custom y-axis tick labels
plt.ylabel('Classifiers')
plt.xlabel('Accuracy (%)')
# plt.title('Comparison of Classifiers Accuracy')
plt.yticks(index, classifiers)

# Add legend with classifier names (the label is already in the bars)
# plt.legend(title='Classifiers')

# Display the plot
plt.tight_layout()
plt.show()
