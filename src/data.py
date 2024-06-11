import os
from typing import List
import numpy as np
from sklearn import preprocessing

def load_data(data_dir: str, window_size: int, horizon: int, stride: int, scaling: str = "S", valid_ratio: float = 0.2):
    train_data = np.load(os.path.join(data_dir, "train_data.npy")) # [n_timesteps, dim]
    test_data = np.load(os.path.join(data_dir, "valid_data.npy")) # [n_timesteps, dim]
    test_periods = np.load(os.path.join(data_dir, "valid_periods.npy")) # [n_periods]
    test_labels = np.load(os.path.join(data_dir, "valid_labels.npy")) # [n_periods]
    
    # devide data into subsequences
    n_valid_data = int(len(train_data) * valid_ratio)
    train_data, valid_data = train_data[:-n_valid_data], train_data[-n_valid_data:]
    train_data, valid_data, test_data = map(lambda data: transform_data(data, scaling), [train_data, valid_data, test_data])
    train_data, valid_data, test_data = map(lambda data: devide_data_into_subsequences(data, window_size + horizon, stride), [train_data, valid_data, test_data])

    # divide data into inputs and outputs
    train_inputs = train_data[:, :window_size] # [n_subsequences, window_size, dim]
    train_outputs = train_data[:, window_size:] # [n_subsequences, horizon, dim]
    valid_inputs = valid_data[:, :window_size] # [n_subsequences, window_size, dim]
    valid_outputs = valid_data[:, window_size:] # [n_subsequences, horizon, dim]
    test_inputs = test_data[:, :window_size] # [n_subsequences, window_size, dim]
    test_outputs = test_data[:, window_size:] # [n_subsequences, horizon, dim]

    return train_inputs, train_outputs, valid_inputs, valid_outputs, test_inputs, test_outputs, test_periods, test_labels


def transform_data(data: List[np.ndarray], scaling: str):
    scaling_dict = {"S": preprocessing.StandardScaler(), "M": preprocessing.MinMaxScaler(), "R": preprocessing.RobustScaler(), "N": preprocessing.Normalizer()}
    scaler = scaling_dict[scaling]
    scaler.fit(data[0])
    transform_data = [scaler.transform(d) for d in data]
    return transform_data


def devide_data_into_subsequences(data: np.ndarray, window_size: int, stride: int):
    n_timesteps, dim = data.shape
    n_subsequences = (n_timesteps - window_size) // stride + 1
    subsequences = np.zeros((n_subsequences, window_size, dim))
    for i in range(n_subsequences):
        start = i * stride
        end = start + window_size
        subsequences[i] = data[start:end]
    return subsequences # [n_subsequences, window_size, dim]