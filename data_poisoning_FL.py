import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef

# Split the dataset into non-IID subsets for devices
def create_non_iid_datasets(X, y, num_devices):
    subsets_X = np.array_split(X, num_devices)
    subsets_y = np.array_split(y, num_devices)
    return subsets_X, subsets_y

class NN(nn.Module):
    def __init__(self, input_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Training and evaluation functions
def local_model_training(model, data, num_epochs, lr):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    for epoch in range(num_epochs):
        for inputs, labels in data:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def evaluate_model(model, data):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in data:
            outputs = model(inputs)
            predicted = outputs.round()
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())
            acc = accuracy_score(y_true, y_pred)
            precison =precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1score = f1_score(y_true, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_true, y_pred)
            except ValueError:
                auc = None
            mcc = matthews_corrcoef(y_true, y_pred)
    return acc, precison, recall, f1score,  auc,  mcc

# Function to introduce data poisoning attack
def poison_data(X, y, fraction=0.5):
    num_poisoned = int(len(y) * fraction)
    poisoned_indices = np.random.choice(len(y), num_poisoned, replace=False)
    y_poisoned = y.clone()
    y_poisoned[poisoned_indices] = 1 - y_poisoned[poisoned_indices]  # Flip labels
    return X, y_poisoned

# Implementing the trimmed mean aggregation
def trimmed_mean_aggregation(updates, trim_ratio):
    n = len(updates)
    k = int(n * trim_ratio)
    sorted_updates = sorted(updates, key=lambda update: np.linalg.norm(update - np.mean(updates, axis=0)))
    return np.mean(sorted_updates[k:n - k], axis=0)


def FL(subsets_X, subsets_y, F, num_devices, num_rounds, num_epochs, num_attackers, trim_ratio):
    input_size = subsets_X[0].shape[1]
    global_model = NN(input_size)

    for round_num in range(num_rounds):
        local_weights = []

        for device in range(num_devices):
            local_model = NN(input_size)
            local_model.load_state_dict(global_model.state_dict())
            X_train = torch.tensor(subsets_X[device], dtype=torch.float32)
            y_train = torch.tensor(subsets_y[device], dtype=torch.float32).view(-1, 1)
            if device == num_attackers:
                X_train, y_train = poison_data(X_train, y_train)
            local_model_training(local_model, [(X_train, y_train)], num_epochs, lr)
            local_weights.append({k: v.clone() for k, v in local_model.state_dict().items()})

        # Convert local weights to numpy arrays for aggregation
        local_weights_np = []
        for lw in local_weights:
            lw_np = {}
            for k, v in lw.items():
                lw_np[k] = v.numpy()
            local_weights_np.append(lw_np)

        # Aggregate local model weights using trimmed mean
        global_weights = {}
        for k in local_weights_np[0].keys():
            updates = np.array([lw_np[k] for lw_np in local_weights_np])
            global_weights[k] = torch.tensor(trimmed_mean_aggregation(updates, trim_ratio))

        global_model.load_state_dict(global_weights)

    return global_model

if __name__ == '__main__':
    pd.options.mode.chained_assignment = None  # default='warn'
    data = fetch_openml(data_id=1590, as_frame=True)
    df = data.frame
    df['income'] = df['class'].apply(lambda x: 1 if x == ' >50K' else 0)
    df = df.drop(columns='class')
    df = pd.get_dummies(df)
    df = df.dropna()
    # Split data into features and labels
    F = df.drop(columns='income').values
    y = df['income'].values
    scaler = StandardScaler()
    F = scaler.fit_transform(F)

    # Parameter settings
    num_devices = 10
    num_attackers = 2
    num_rounds = 30
    num_epochs = 3
    lr = 0.01 # learning rate
    trim_ratio = 0.2

    subsets_X, subsets_y = create_non_iid_datasets(F, y, num_devices)
    global_model = FL(subsets_X, subsets_y, F, num_devices, num_rounds, num_epochs, num_attackers, trim_ratio)

    local_models = []
    for device in range(num_devices):
        local_model = NN(subsets_X[0].shape[1])
        local_model.load_state_dict(global_model.state_dict())

        X_train = torch.tensor(subsets_X[device], dtype=torch.float32)
        y_train = torch.tensor(subsets_y[device], dtype=torch.float32).view(-1, 1)

        local_model_training(local_model, [(X_train, y_train)], num_epochs, lr)
        local_models.append(local_model)

    # Evaluate local models
    for device in range(num_devices):
        X_test = torch.tensor(subsets_X[device], dtype=torch.float32)
        y_test = torch.tensor(subsets_y[device], dtype=torch.float32).view(-1, 1)
        acc, precison, recall, f1score,  auc,  mcc = evaluate_model(local_models[device],
                                                                             [(X_test, y_test)])
        print(f'Device {device + 1}, Local Model Test Accuracy: {acc:.4f}, Precision: {precison:.4f}, Recall: {recall:.4f}, F1-Score: {f1score:.4f}, MCC: {mcc:.4f}')
