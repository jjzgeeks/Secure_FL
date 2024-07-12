import torch
import copy
import random
import statistics
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import  LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
from scipy.io import savemat


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# Split the dataset into non-IID and poisoning subsets for devices
def create_non_iid_poison_datasets(train_dataset, num_devices, num_attackers, poison_ratio):
    devices = [[] for _ in range(num_devices)]
    X_train = train_dataset.dataset.tensors[0][train_dataset.indices]
    y_train = train_dataset.dataset.tensors[1][train_dataset.indices]

    label_0_indices = np.where(y_train == 0)[0]
    label_1_indices = np.where(y_train == 1)[0]
    np.random.shuffle(label_0_indices)
    np.random.shuffle(label_1_indices)

    # Distribute indices into devices
    for i in range(num_devices):
        if i < 2:
            # devices with only label 0
            devices[i] = label_0_indices[i::num_devices].tolist()
        elif i <= 3:
            # devices with only label 1
            devices[i] = label_1_indices[i::num_devices].tolist()
        else:
            # devices with both labels
            mixed_indices = np.concatenate((label_0_indices[i::num_devices], label_1_indices[i::num_devices]))
            np.random.shuffle(mixed_indices)
            devices[i] = mixed_indices.tolist()

    if num_attackers < num_devices:
       poisoned_devices = list(range(num_attackers))
       for idx in poisoned_devices:
           num_poisoned_samples = int(poison_ratio * len(devices[idx]))  # Poison (poison_ratio * 100)% of the group's data
           poisoned_idx = np.random.choice(devices[idx], num_poisoned_samples, replace=False)
           y_train[poisoned_idx] = 1 - y_train[poisoned_idx]  # Flip the labels of poisoned samples

    devices_datasets = []
    for device_idx in devices:
        device_dataset = TensorDataset(X_train[device_idx], y_train[device_idx])
        devices_datasets.append(device_dataset)
    return devices_datasets



"Define the training model"
class NN(nn.Module):
    def __init__(self, input_dim):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def local_model_update(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    e_loss = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        e_loss.append(train_loss)
    total_loss = sum(e_loss) / len(e_loss)
    return model.state_dict(), total_loss


""" Aggregation algorithms """
def FedAvg(local_models):
    avg_state_dict = {}
    for key in local_models[0].keys():
        avg_state_dict[key] = torch.stack([state_dict[key].float() for state_dict in local_models]).mean(dim=0)
    return avg_state_dict

def Trimmed_mean(local_models, trim_ratio):
    trim_k = int(len(local_models) * trim_ratio)
    avg_state_dict = {}
    for key in local_models[0].keys():
        stacked_weights = torch.stack([state_dict[key].float() for state_dict in local_models])
        sorted_weights, _ = torch.sort(stacked_weights, dim=0)
        trimmed_weights = sorted_weights[trim_k:-trim_k]
        avg_state_dict[key] = trimmed_weights.mean(dim=0)
    return avg_state_dict

def Krum(local_models, f):
    n = len(local_models)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance = 0
            for key in local_models[0].keys():
                distance += torch.sum((local_models[i][key] - local_models[j][key]) ** 2).item()
            distances[i, j] = distance
            distances[j, i] = distance
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = np.sum(np.sort(distances[i])[:n - f - 2])
    print(scores)
    print(np.argmin(scores))
    return local_models[np.argmin(scores)]

def Multi_Krum(local_models, f):
    n = len(local_models)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance = 0
            for key in local_models[0].keys():
                distance += torch.sum((local_models[i][key] - local_models[j][key]) ** 2).item()
            distances[i, j] = distance
            distances[j, i] = distance

    scores = np.zeros(n)
    for i in range(n):
        scores[i] = np.sum(np.sort(distances[i])[:n - f - 2])

    sorted_indices = np.argsort(scores)
    selected_indices = sorted_indices[:n - f - 2]

    aggregated_state_dict = {}
    for key in local_models[0].keys():
        aggregated_state_dict[key] = torch.mean(torch.stack([local_models[i][key] for i in selected_indices]), dim=0)
    return aggregated_state_dict

def Median(local_models):
    aggregated_state_dict = {}
    for key in local_models[0].keys():
        stacked_tensors = torch.stack([state_dict[key] for state_dict in local_models])
        aggregated_state_dict[key] = torch.median(stacked_tensors, dim=0).values
    return aggregated_state_dict


def global_model_testing(model, test_loader, criterion):
    model.eval()
    test_loss, correct = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).squeeze()
            test_loss += criterion(outputs, y_batch.float()).item()
            preds = torch.round(torch.sigmoid(outputs))
            correct += (preds == y_batch).sum().item()
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.numpy().flatten().astype(int))
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    #acc = accuracy_score(y_true, y_pred)
    precison = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    return test_loss, accuracy, precison, recall, f1score, auc, mcc


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None  # default='warn'
    data = fetch_openml(data_id=1590, as_frame=True)
    df = data.frame
    df = df.apply(LabelEncoder().fit_transform)
    X = df.drop(columns=['class'])
    y = df['class']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    num_samples, num_features = X.shape # (48842, 14)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    # Split the data into training (80%) and testing (20%)
    train_size = int(0.8 * len(X_tensor))
    test_size = len(X_tensor) - train_size
    train_dataset, test_dataset = random_split(TensorDataset(X_tensor, y_tensor), [train_size, test_size])

    # Parameter settings
    num_devices = 8
    #num_attackers = 9 # num_attackers > num_deices represents no data poisoning
    num_attackers = 2
    poison_ratio = 0.6
    num_rounds = 30
    num_epochs = 3
    lr = 0.05 # learning rate
    trim_ratio = 0.4
    batch_size = 64
    Aggregation_set = ['FedAvg', 'Trimmed_mean', 'Krum', 'Multi-Krum', 'Median']
    Agg = Aggregation_set[5]

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    devices_datasets = create_non_iid_poison_datasets(train_dataset, num_devices, num_attackers, poison_ratio)
    global_model = NN(num_features)
    criterion = nn.BCEWithLogitsLoss()

    Train_loss, Test_loss = [], []
    ACC, Precison, Recall, F1_score, AUC, MCC = [],[],[],[],[],[]
    for round in range(num_rounds):
        print(f'Round {round + 1} starting...')
        local_models, local_losses = [], []
        for device_idx, device_dataset in enumerate(devices_datasets):
            local_model = NN(num_features)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = optim.Adam(local_model.parameters(), lr=lr)
            train_loader = DataLoader(device_dataset, batch_size=batch_size, shuffle=True)
            local_model, local_loss = local_model_update(local_model, train_loader, criterion, optimizer, num_epochs)
            local_models.append(local_model)
            local_losses.append(local_loss)

        """Selecting an aggregation algorithm """
        if Agg == 'FedAvg':
            global_model.load_state_dict(FedAvg(local_models)) # Fed average aggregation
        elif Agg == 'Trimmed_mean':
            global_model.load_state_dict(Trimmed_mean(local_models, trim_ratio)) # Trimmed mean aggregation
        elif Agg == 'Krum':
            global_model.load_state_dict(Krum(local_models, num_attackers)) # Krum aggregation
            #global_model.load_state_dict(local_models[num_devices-1])
        elif Agg == 'Multi-Krum':
            global_model.load_state_dict(Multi_Krum(local_models, num_attackers))  # Multi Krum aggregation
        else:
            global_model.load_state_dict(Median(local_models))

        train_loss = sum(local_losses) / len(local_losses)
        Train_loss.append(train_loss)
        # global model testing
        test_loss, test_accuracy, precison, recall, f1score, auc, mcc = global_model_testing(global_model, test_loader, criterion)
        print(f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        Test_loss.append(test_loss)
        ACC.append(test_accuracy)
        Precison.append(precison)
        F1_score.append(f1score)
        AUC.append(auc)
        MCC.append(mcc)
    Precison_mean, Precison_std = statistics.mean(Precison), statistics.stdev(Precison)
    F1_score_mean, F1_score_std = statistics.mean(F1_score), statistics.stdev(F1_score)
    AUC_mean, AUC_std = statistics.mean(AUC), statistics.stdev(AUC)
    MCC_mean, MCC_std = statistics.mean(MCC), statistics.stdev(MCC)
    print(f'Precison: {Precison_mean, Precison_std}')
    print(f'F1_score: {F1_score_mean, F1_score_std}')
    print(f'AUC: {AUC_mean, AUC_std}')
    print(f'MCC: {MCC_mean, MCC_std}')

    metrics_mean_set = [Precison_mean, F1_score_mean, AUC_mean, MCC_mean]
    metrics_std_set = [Precison_std, F1_score_std, AUC_std, MCC_std]
    savemat("./FL_data_poisoning_{}_{}_{}_{}.mat".format(Agg, num_rounds, num_devices, num_attackers), {"Accuracy": ACC,
    "Metrics_mean_set": metrics_mean_set, "Metrics_std_set": metrics_std_set})
    print('Federated Learning Training Done!')

    rounds = list(range(num_rounds))
    plt.figure()
    plt.plot(rounds, Train_loss, label='Train loss')
    plt.plot(rounds, Test_loss, label='Test loss')
    plt.xlabel('Communication round')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig('./Loss_{}_{}_{}_{}.png'.format(Agg,num_rounds, num_devices, num_attackers))

    plt.figure()
    plt.plot(rounds, ACC, label='Train loss')
    plt.xlabel('Communication round')
    plt.ylabel('Test accuracy')
    #plt.title('Global Model Test Accuracy Without Defense')
    plt.savefig('./Accuracy_{}_{}_{}_{}.png'.format(Agg, num_rounds, num_devices, num_attackers))
    plt.show()




