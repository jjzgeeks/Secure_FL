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

# Split the dataset into non-IID for devices
def create_non_iid_poison_datasets(train_dataset, num_devices):
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

def add_noise(tensor, mean=0.0, std=0.1):
    noise = torch.normal(mean, std, tensor.size())
    return tensor + noise



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


def local_model_testing(model, data_loader, criterion):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch).squeeze()
            loss += criterion(outputs, y_batch.float()).item()
            preds = torch.round(torch.sigmoid(outputs))
            correct += (preds == y_batch).sum().item()
    accuracy = correct / len(data_loader.dataset)
    return accuracy


def FedAda(global_model, local_models, local_weights):
    """
    Aggregates the local models using adaptive weights.
    """
    global_model_state_dict = global_model.state_dict()
    for key in global_model_state_dict.keys():
        global_model_state_dict[key] = torch.sum(
            torch.stack([local_weights[i] * local_models[i][key] for i in range(len(local_models))]), dim=0
        )
    global_model.load_state_dict(global_model_state_dict)
    return global_model




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
    num_rounds = 30
    num_epochs = 3
    lr = 0.05 # learning rate
    batch_size = 64


    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    devices_datasets = create_non_iid_poison_datasets(train_dataset, num_devices)
    global_model = NN(num_features)
    global_state = global_model.state_dict()
    criterion = nn.BCEWithLogitsLoss()

    Train_loss, Test_loss = [], []
    ACC, Precison, Recall, F1_score, AUC, MCC = [],[],[],[],[],[]
    for round in range(num_rounds):
        print(f'Round {round + 1} starting...')
        device_state_dicts, local_losses, local_accuracies = [], [], []
        for device_idx, device_dataset in enumerate(devices_datasets):
            local_model = NN(num_features)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = optim.Adam(local_model.parameters(), lr=lr)
            train_loader = DataLoader(device_dataset, batch_size=batch_size, shuffle=True)

            device_state_dict, local_loss = local_model_update(local_model, train_loader, criterion, optimizer, num_epochs)
            device_state_dicts.append(device_state_dict)
            local_losses.append(local_loss)

            local_accuracy = local_model_testing(local_model, train_loader, criterion)
            local_accuracies.append(local_accuracy)

        # Normalize device accuracies to obtain weights
        local_weights = [acc / sum(local_accuracies) for acc in local_accuracies]
        global_model = FedAda(global_model, device_state_dicts, local_weights)

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
    savemat("./FedAda_{}_{}.mat".format(num_rounds, num_devices), {"Accuracy": ACC,
    "Metrics_mean_set": metrics_mean_set, "Metrics_std_set": metrics_std_set})
    print('Federated Learning Training Done!')


    rounds = list(range(num_rounds))
    plt.figure()
    plt.plot(rounds, Train_loss, label='Train loss')
    plt.plot(rounds, Test_loss, label='Test loss')
    plt.xlabel('Communication round')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig('./FedAda_Loss_{}_{}.png'.format(num_rounds, num_devices))

    plt.figure()
    plt.plot(rounds, ACC, label='Train loss')
    plt.xlabel('Communication round')
    plt.ylabel('Test accuracy')
    #plt.title('Global Model Test Accuracy')
    plt.savefig('./FedAda_Accuracy_{}_{}.png'.format(num_rounds, num_devices))
    plt.show()
