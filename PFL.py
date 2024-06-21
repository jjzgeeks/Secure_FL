import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler



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


def model_testing(model, data):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data:
            outputs = model(inputs)
            predicted = outputs.round()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def FL(subsets_X, subsets_y, X, num_devices, num_rounds, num_epochs, lr):
    input_size = subsets_X[0].shape[1]
    global_model = NN(input_size)
    test_accuracy_set = []

    for round_num in range(num_rounds):
        local_weights = []
        for device in range(num_devices):
            local_model = NN(input_size)
            local_model.load_state_dict(global_model.state_dict())
            X_train = torch.tensor(subsets_X[device], dtype=torch.float32)
            y_train = torch.tensor(subsets_y[device], dtype=torch.float32).view(-1, 1)
            local_model_training(local_model, [(X_train, y_train)], num_epochs, lr)
            local_weights.append(local_model.state_dict())
        global_weights = global_model.state_dict()
        for key in global_weights.keys():
            global_weights[key] = torch.stack([local_weights[i][key] for i in range(num_devices)], 0).mean(0)
        global_model.load_state_dict(global_weights)
        X_test = torch.tensor(X, dtype=torch.float32)
        y_test = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        test_accuracy = model_testing(global_model, [(X_test, y_test)])
        test_accuracy_set.append(test_accuracy)
        print(f'Round {round_num + 1}, test accuracy: {test_accuracy:.5f}')
    return global_model, test_accuracy_set






if __name__ == '__main__':
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
    num_devices = 10
    num_rounds = 30
    num_epochs = 3
    lr = 0.01

    subsets_X, subsets_y = create_non_iid_datasets(F, y, num_devices)

    global_model, test_accuracy_set = FL(subsets_X, subsets_y, F, num_devices, num_rounds, num_epochs, lr)
    personalized_models = []

    for device in range(num_devices):
        personalized_model = NN(subsets_X[0].shape[1])
        personalized_model.load_state_dict(global_model.state_dict())
        X_train = torch.tensor(subsets_X[device], dtype=torch.float32)
        y_train = torch.tensor(subsets_y[device], dtype=torch.float32).view(-1, 1)
        local_model_training(personalized_model, [(X_train, y_train)], num_epochs, lr)
        personalized_models.append(personalized_model)

    plt.figure()
    plt.plot(list(range(1, num_rounds+1)), test_accuracy_set)
    plt.xlabel('Communication rounds')
    plt.ylabel('Test accuracy of global model')
    plt.title('Personalized federated learning')
    plt.savefig('./Test_accuracy_{}_{}.png'.format(num_devices, num_rounds))
    plt.show()


    # Testing personalized models
    for device in range(num_devices):
        X_test = torch.tensor(subsets_X[device], dtype=torch.float32)
        y_test = torch.tensor(subsets_y[device], dtype=torch.float32).view(-1, 1)
        test_accuracy = model_testing(personalized_models[device], [(X_test, y_test)])
        print(f'Device {device + 1}, personalized FL model test accuracy: {test_accuracy:.5f}')
