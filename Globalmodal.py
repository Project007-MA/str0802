import torch
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

# Function to perform FedAvg
def fed_avg(models):
    global_model = copy.deepcopy(models[0])  # Initialize with one local model
    global_dict = global_model.state_dict()
    
    # Average the model weights
    for key in global_dict.keys():
        global_dict[key] = torch.mean(
            torch.stack([models[i].state_dict()[key] for i in range(len(models))]), dim=0
        )
    
    global_model.load_state_dict(global_dict)
    return global_model

# Generate dummy datasets for local training
def get_dataset():
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100, 1), dtype=torch.float32)
    return TensorDataset(X, y)

dataset_1 = get_dataset()
dataset_2 = get_dataset()

dataloader_1 = DataLoader(dataset_1, batch_size=10, shuffle=True)
dataloader_2 = DataLoader(dataset_2, batch_size=10, shuffle=True)

# Training function
def train_model(model, dataloader, epochs=5):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    history = {'loss': []}
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            X_batch, y_batch = batch
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        history['loss'].append(total_loss / len(dataloader))
    return history

# Create two local models
local_model_1 = SimpleModel()
local_model_2 = SimpleModel()

# Train both models on their respective datasets
history_1 = train_model(local_model_1, dataloader_1)
history_2 = train_model(local_model_2, dataloader_2)

# Perform Federated Averaging
models = [local_model_1, local_model_2]
global_model = fed_avg(models)

# Evaluate performance on a test dataset
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            X_batch, y_batch = batch
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss

# Generate a test dataset
test_dataset = get_dataset()
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Evaluate the global model
accuracy, avg_loss = evaluate_model(global_model, test_dataloader)
print(f"Global Model Performance: Accuracy = {accuracy:.4f}, Loss = {avg_loss:.4f}")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(history_1['loss'], label='Local Model 1 Loss')
plt.plot(history_2['loss'], label='Local Model 2 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss for Local Models')
plt.legend()
plt.show()

# Plot global model performance metrics
epochs = range(1, len(history_1['loss']) + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs, [avg_loss] * len(epochs), label='Global Model Loss', linestyle='dashed')
plt.axhline(y=accuracy, color='r', linestyle='-', label='Global Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.title('Global Model Performance')
plt.legend()
plt.show()
