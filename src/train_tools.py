import os, pathlib
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import metrics
import plots

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 logger,
                 result_dir: pathlib.Path,
                 train_inputs: np.ndarray,
                 train_outputs: np.ndarray,
                 valid_inputs: np.ndarray,
                 valid_outputs: np.ndarray,
                 criterion: str = "MSELoss",
                 batch_size: int = 32,
                 n_epochs: int = 100,
                 optimizer: str = "Adam",
                 learning_rate: float = 1e-3,
                 max_grad_norm: float = 1.0,
                 n_evaluate_every: int = 1,
                 cutoff_time_window: int = 12,
                 smoothed_window: int = 90):
        self.model = model
        self.device = device
        self.logger = logger
        self.result_dir = result_dir
        self.criterion = getattr(nn, criterion)(reduction="none")
        self.n_timesteps = len(train_inputs)
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm
        self.n_evaluate_every = n_evaluate_every
        self.cutoff_time_window = cutoff_time_window
        self.smoothed_window = smoothed_window

        # prepare data loader
        train_tensor = torch.utils.data.TensorDataset(torch.from_numpy(train_inputs).float(), torch.from_numpy(train_outputs).float())
        valid_tensor = torch.utils.data.TensorDataset(torch.from_numpy(valid_inputs).float(), torch.from_numpy(valid_outputs).float())
        self.train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(valid_tensor, batch_size=batch_size, shuffle=False)

        # prepare optimizer
        self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), lr=learning_rate)
        self.alpha, self.train_period_class, self.train_class_probs = None, None, None


    def save(self, epoch: int, model_type: str = "best"):
        data = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "alpha": self.alpha,
            "train_period_class": self.train_period_class,
            "train_class_probs": self.train_class_probs
        }
        torch.save(data, os.path.join(self.result_dir, f"{model_type}_model.pt"))


    def load(self, model_type: str = "best"):
        data = torch.load(os.path.join(self.result_dir, f"{model_type}_model.pt"))
        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])
        return data["epoch"]


    def train(self, verbose: bool = True):
        self.model.train()
        best_valid_loss = 1e9
        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0.
            train_attn = []
            for inputs, outputs in self.train_loader:
                inputs, outputs = inputs.to(self.device), outputs.to(self.device)
                self.optimizer.zero_grad()
                preds, attn = self.model(inputs, return_attn=True)
                loss = self.criterion(preds, outputs).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                epoch_loss += loss.item()
                train_attn.append(attn[:,0])

            epoch_loss /= len(self.train_loader)
            self.logger.log_metric("train_loss", epoch_loss, epoch)
            if verbose:
                print(f"Epoch {epoch+1}/{self.n_epochs} - Loss: {epoch_loss:.4f}")

            if (epoch + 1) % self.n_evaluate_every == 0:
                self.model.eval()
                epoch_loss = 0.
                for inputs, outputs in self.valid_loader:
                    inputs, outputs = inputs.to(self.device), outputs.to(self.device)
                    preds = self.model(inputs)
                    loss = self.criterion(preds, outputs).mean()
                    epoch_loss += loss.item()

                epoch_loss /= len(self.valid_loader)
                self.logger.log_metric("valid_loss", loss.item(), epoch)
                if verbose:
                    print(f"=====> Validation Loss: {epoch_loss:.4f}")
                if epoch_loss < best_valid_loss:
                    best_valid_loss = epoch_loss
                    # compute dirichlet parameters
                    train_attn = torch.cat(train_attn, dim=0)
                    self.alpha = metrics.maximum_likelihood_estimation_of_Dirichlet_distribution(train_attn)
                    self.save(epoch, "best")
                    self.train_period_class, self.train_class_probs = metrics.compute_multinomial_probabilities(train_attn, self.model.window_size, self.cutoff_time_window, self.smoothed_window)
                    if verbose:
                        print(f"=====> Best model saved at epoch {epoch+1} - Loss: {best_valid_loss:.4f}")

        
    def test(self, test_inputs: np.ndarray, test_outputs: np.ndarray, test_labels: np.ndarray, test_periods: Optional[np.ndarray] = None, verbose: bool = False):
        self.model.eval()
        test_tensor = torch.utils.data.TensorDataset(torch.from_numpy(test_inputs).float(), torch.from_numpy(test_outputs).float())
        test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=1, shuffle=False)

        test_attn, test_loss = [], []
        for inputs, outputs in test_loader:
            inputs, outputs = inputs.to(self.device), outputs.to(self.device)
            preds, attn = self.model(inputs, return_attn=True)
            test_loss.append(self.criterion(preds, outputs).mean(dim=(1,2)))
            test_attn.append(attn[:,0])
        test_loss = torch.cat(test_loss, dim=0)
        test_attn = torch.cat(test_attn, dim=0)

        # compute dirichlet anomalous score
        dirichlet_nll = - metrics.compute_dirichlet_log_density(test_attn, self.alpha)

        # compute estimated periods
        estimated_periods = metrics.compute_highest_attention_time_excluding_cutoff_time_window(test_attn, self.model.window_size, self.cutoff_time_window) # [n_timesteps]
        estimated_periods = metrics.median_smoothing(estimated_periods, self.smoothed_window) # [n_timesteps]

        # compute anomaly scores
        if test_periods is not None:
            start_period_idx = 0
            residual = self.model.window_size
            while residual > 0:
                residual -= test_periods[start_period_idx]
                start_period_idx += 1
            start_data_idx = - residual

            end_period_idx = len(test_periods) - 1
            residual = self.model.horizon - 1
            while residual > 0:
                residual -= test_periods[end_period_idx]
                end_period_idx -= 1
            end_period_idx += 1
            end_data_idx = len(test_outputs) - start_data_idx - residual

            anomalous_statistics = torch.stack([test_loss, dirichlet_nll, estimated_periods], dim=1)[start_data_idx:end_data_idx] # [n_timesteps, 3]
            test_periods = test_periods[start_period_idx:end_period_idx] # [n_periods]
            test_labels = test_labels[start_period_idx:end_period_idx] # [n_periods]
            anomalous_statistics = metrics.aggregate_statistics_for_each_periods(anomalous_statistics, test_periods) # [n_periods, 3]
            test_loss, dirichlet_nll, estimated_periods = anomalous_statistics.transpose(0, 1) # [n_periods]
            period_nll = metrics.compute_negative_log_likelihood_multinomial(estimated_periods, self.train_period_class, self.train_class_probs) # [n_periods]
            test_loss, dirichlet_nll, estimated_periods, period_nll = map(lambda x: x.detach().cpu().numpy(), [test_loss, dirichlet_nll, estimated_periods, period_nll])

            for name, score in zip(["pred", "dirichlet", "period"], [test_loss, dirichlet_nll, period_nll]):
                plots.plot_precision_recall_curve(self.logger, self.result_dir, test_labels, score, name)
                plots.plot_roc_curve(self.logger, self.result_dir, test_labels, score, name)

            plots.plot_period_transition(self.logger, self.result_dir, test_periods, estimated_periods)