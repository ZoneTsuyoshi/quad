import math
import numpy as np


def generate_quasi_periodic_five(dim: int, n_periods: int, base_period: int, period_fluctuation: float, cutoff_period_difference: int, n_waves: int, anomaly_period_bias: int, anomaly_continuous_probability: float, std: float, anomaly_probability: float, seed: int) -> np.ndarray:
    np.random.seed(seed)
    
    amplitudes = 1 + 5*np.random.random([dim, n_waves]) #(dim,n_waves)
    phases = 2*math.pi*np.random.random([dim, n_waves]) #(dim,n_waves)

    times = np.arange(base_period).reshape(-1,1,1).repeat(dim, axis=1).repeat(n_waves, axis=2) #(period,dim,nw)
    data = np.sum(amplitudes * np.cos(2*math.pi*times/base_period*np.arange(1,n_waves+1) + phases), axis=2) #(period,dim)
    period_storage = [base_period]
    anomaly_flags = [0.]
    bias_sign = 2 * int(np.random.rand(1) < 0.5) - 1
    ano_prob = anomaly_probability

    for _ in range(n_periods-1):
        anomaly_on = np.random.rand(1) < ano_prob
        anomaly_flags.append(float(anomaly_on))
        period = base_period + np.clip(np.random.normal(0, period_fluctuation), -cutoff_period_difference, cutoff_period_difference)
        if anomaly_on:
            period += bias_sign * anomaly_period_bias
            ano_prob = anomaly_continuous_probability
        else:
            bias_sign = 2 * int(np.random.rand(1) < 0.5) - 1
            ano_prob = anomaly_probability
        times = np.arange(period).reshape(-1,1,1).repeat(dim, axis=1).repeat(n_waves, axis=2) #(period,dim,nw)
        new_data = np.sum(amplitudes * np.cos(2*math.pi*times/period*np.arange(1,n_waves+1) + phases), axis=2) #(period,dim)
        data = np.concatenate([data, new_data], axis=0) #(T,dim)
        period_storage.append(period)
        
    obs = data + data * np.absolute(np.random.normal(0, std, size=data.shape))
    period_storage = np.array(period_storage)
    anomaly_flags = np.array(anomaly_flags)
    return obs, period_storage, anomaly_flags