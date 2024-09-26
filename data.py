import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from importlib import import_module
from torch.utils.data import TensorDataset, DataLoader


def get_dataset(name, datasets_folder='datasets', seed=0, test_ratio=0.2, calibration_ratio=0.2):
    loader = import_module(f'.{name}.load', package=f'{datasets_folder}')
    X, Y = loader.load()

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    idx_train, idx_test = train_test_split(np.arange(X.shape[0]), test_size=test_ratio, random_state=seed)
    X_train, X_test, Y_train, Y_test = X[idx_train], X[idx_test], Y[idx_train], Y[idx_test]

    idx_train, idx_calibration = train_test_split(np.arange(X_train.shape[0]), test_size=calibration_ratio, random_state=seed)
    X_train_final, X_calibration = X_train[idx_train], X_train[idx_calibration]
    Y_train_final, Y_calibration = Y_train[idx_train], Y_train[idx_calibration]

    return X_train_final, Y_train_final, X_calibration, Y_calibration, X_test, Y_test


def standardization_data(train, test, rel_indices):
    sx = StandardScaler()
    sx.fit(train[rel_indices])
    train = sx.transform(train)
    test = sx.transform(test)
    return train, test


def create_train_test_loaders(X_train, X_test, y_train, y_test, batch_size):
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader


def get_concatenate_pred_true_and_coefficients(mean_model, uncertainty_model, X, y, batch_size=32):
    mean_model.eval()
    uncertainty_model.eval()

    data = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(data, batch_size=batch_size)

    pred_list = []
    uncertainty_coefficients_list = []
    true_list = []

    with torch.no_grad():
        for X, y in loader:
            pred = mean_model(X)
            X_concat = torch.cat((X, pred), dim=1)
            pred_list.append(pred.detach().cpu().numpy())

            uncertainty_coeffs = uncertainty_model(X_concat)
            uncertainty_coefficients_list.append(uncertainty_coeffs.detach().cpu().numpy())
            true_list.append(y.detach().cpu().numpy())

    return (np.concatenate(pred_list, axis=0),
            np.concatenate(uncertainty_coefficients_list, axis=0),
            np.concatenate(true_list, axis=0))
