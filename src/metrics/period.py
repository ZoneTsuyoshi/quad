import torch

@torch.no_grad()
def compute_highest_attention_time_excluding_cutoff_time_window(x: torch.Tensor, original_window_size: int, cutoff_time_window: int):
    current_window_size = x.shape[1]
    window_difference = original_window_size - current_window_size
    half_window_difference = window_difference // 2
    highest_reversed_time = x[:,:-cutoff_time_window].max(dim=1)[1]
    highest_reversed_time = highest_reversed_time + half_window_difference + cutoff_time_window + 1
    return highest_reversed_time

@torch.no_grad()
def median_smoothing(x: torch.Tensor, smoothed_window: int = 100):
    smoothed_x = torch.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - smoothed_window // 2)
        end = min(len(x), i + smoothed_window // 2)
        smoothed_x[i] = x[start:end].median()
    return smoothed_x

@torch.no_grad()
def compute_multinomial_probabilities(x: torch.Tensor, original_window_size: int, cutoff_time_window: int, smoothed_window: int):
    estimated_periods = compute_highest_attention_time_excluding_cutoff_time_window(x, original_window_size, cutoff_time_window) # [n_timesteps]
    estimated_periods = median_smoothing(estimated_periods, smoothed_window) # [n_timesteps]

    # compute emerging probs of each periods
    periods, counts = torch.unique(estimated_periods, return_counts=True)
    return periods, counts / counts.sum()

@torch.no_grad()
def compute_negative_log_likelihood_multinomial(x: torch.Tensor, periods: torch.Tensor, class_probs: torch.Tensor):
    """
    Args:
        x: [n_timesteps]
        periods: [n_period_class]
        class_probs: [n_period_class]
    """
    negative_log_likelihood = torch.zeros_like(x, dtype=torch.float32) + 1e+6
    for i, p in enumerate(periods):
        negative_log_likelihood[x==p] = -torch.log(class_probs[i])
    return negative_log_likelihood

@torch.no_grad()
def aggregate_statistics_for_each_periods(x: torch.Tensor, periods: torch.Tensor):
    n_periods = len(periods)
    aggregated_statistics = []
    counter = 0
    for p in periods:
        aggregated_statistics.append(x[counter:counter+p].mean(dim=0))
    return torch.stack(aggregated_statistics, dim=0) # [n_periods, 3]