import math
import torch
from torch import optim, nn
from data import create_train_test_loaders, get_concatenate_pred_true_and_coefficients
from plots import plot_points_with_uncertainty_shapes, draw_ellipsoids
from uncertainty_loss import uncertainty_loss, uncertainty_function_at_t, custom_integral
from sklearn.neighbors import KNeighborsRegressor
from scipy.special import softmax as scipy_softmax
import numpy as np


class MeanLearner:

    def __init__(self, epochs, batch_size, model, mean_model_path):
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model_dict_path = mean_model_path

    def fit(self, X_train, X_test, y_train, y_text):
        train_loader, test_loader = create_train_test_loaders(X_train, X_test, y_train, y_text, self.batch_size)
        train_loss, test_loss = [], []

        for i in range(self.epochs):
            self.model.train()
            cur_loss = 0
            for X, y in train_loader:
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.criterion(output, y)
                cur_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            train_loss.append(cur_loss / len(train_loader))

            self.model.eval()
            with torch.no_grad():
                cur_loss = 0
                for X, y in test_loader:
                    output = self.model(X)
                    loss = self.criterion(output, y)
                    cur_loss += loss.item()

                test_loss.append(cur_loss / len(test_loader))

        torch.save(self.model.state_dict(), self.model_dict_path)
        return train_loss, test_loss


class UncertaintyLearner:

    def __init__(self, epochs, batch_size, uncertainty_model, mean_model, lamda, uncertainty_model_path,
                 database_name, seed):
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = uncertainty_loss
        self.uncertainty_model = uncertainty_model
        self.mean_model = mean_model
        self.optimizer = optim.Adam(self.uncertainty_model.parameters(), lr=0.001)
        self.model_dict_path = uncertainty_model_path
        self.lamda = lamda
        self.database_name = database_name
        self.seed = seed
        self.scale = 1

    def fit(self, X_train, X_test, y_train, y_test):
        train_loader, test_loader = create_train_test_loaders(X_train, X_test, y_train, y_test, self.batch_size)
        train_loss, test_loss = [], []
        self.mean_model.eval()

        for i in range(self.epochs):
            print(f"epoch: {i}")
            self.uncertainty_model.train()
            cur_loss = 0
            for X, y in train_loader:
                self.optimizer.zero_grad()
                mean_pred = self.mean_model(X)
                X_concat = torch.cat((X, mean_pred), dim=1)

                coefficients = self.uncertainty_model(X_concat)
                loss = self.criterion(coefficients, y, mean_pred, self.lamda)
                cur_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            train_loss.append(cur_loss / len(train_loader))

            self.uncertainty_model.eval()
            with torch.no_grad():
                cur_loss = 0
                for X, y in test_loader:
                    mean_pred = self.mean_model(X)
                    X_concat = torch.cat((X, mean_pred), dim=1)
                    coefficients = self.uncertainty_model(X_concat)

                    loss = self.criterion(coefficients, y, mean_pred, self.lamda)
                    cur_loss += loss.item()

                test_loss.append(cur_loss / len(test_loader))

                # if i % 10 == 0 or i == self.epochs - 1:
                pred_test, uncertainty_coefficients_test, true_test = get_concatenate_pred_true_and_coefficients(
                    self.mean_model, self.uncertainty_model, X_train[:10], y_train[:10])
                plot_points_with_uncertainty_shapes(true_test, pred_test, uncertainty_coefficients_test, 1,
                                                    self.database_name, self.seed, i,
                                                    f"MVSVRC uncertainty shapes - epoch {i} (training set)")

        torch.save(self.uncertainty_model.state_dict(), self.model_dict_path)
        return train_loss, test_loss

    def calibrate(self, X, y, u=0.9, reset=False):
        mean_pred = self.mean_model(torch.tensor(X))
        X_concat = torch.cat((torch.tensor(X), mean_pred), dim=1)
        coefficients = self.uncertainty_model(X_concat)

        coefficients = coefficients.detach().numpy()

        delta = torch.tensor(y) - mean_pred
        delta_x = delta[:, 0]
        delta_y = delta[:, 1]
        angles = torch.atan2(delta_y, delta_x)

        uncertainty_values = torch.stack([
            uncertainty_function_at_t(coefficients[i], angles[i]) for i in range(len(angles))
        ])

        distance_to_true = torch.norm(delta, dim=1)

        conformity_scores = (distance_to_true - uncertainty_values).clone().detach().numpy()
        sorted_indices = np.argsort(conformity_scores)
        sorted_scores = conformity_scores[sorted_indices]
        sorted_uncertainty_values = uncertainty_values[sorted_indices]
        q_index = int(np.ceil(u * len(sorted_scores))) - 1
        self.scale = (sorted_scores[q_index] + sorted_uncertainty_values[q_index]) / sorted_uncertainty_values[q_index]

    def evaluate_and_plot_shapes(self, X, y):
        mean_pred = self.mean_model(torch.tensor(X))
        X_concat = torch.cat((torch.tensor(X), mean_pred), dim=1)
        coefficients = self.uncertainty_model(X_concat).detach().numpy()

        delta = torch.tensor(y) - mean_pred
        delta_x = delta[:, 0]
        delta_y = delta[:, 1]
        angles = torch.atan2(delta_y, delta_x)

        uncertainty_values = self.scale * torch.stack([
            uncertainty_function_at_t(coefficients[i], angles[i]) for i in range(len(angles))
        ])

        distance_to_true = torch.norm(delta, dim=1)
        accuracy = (uncertainty_values >= distance_to_true).sum().item() / len(angles)

        num_steps = 50
        t_values = torch.linspace(0, 2 * math.pi, num_steps)
        avg_uncertainty_shape_area = custom_integral(coefficients, t_values, self.scale).item() / coefficients.shape[0]

        plot_points_with_uncertainty_shapes(y[:10], mean_pred[:10].detach().numpy(), coefficients[:10],
                                            self.scale, self.database_name, self.seed, -1,
                                            title=f"MVSRC uncertainty shapes - accuracy of {round(accuracy, 3)},"
                                       f"average area of {round(avg_uncertainty_shape_area, 3)} (test set)")

        return accuracy, avg_uncertainty_shape_area

class BaseLearner:
    def __init__(self):
        self.scale = 1.
        self.cov = None

    def fit(self, X, diff, to_force=False, **kwargs):
        self.cov = np.cov(diff.T)
        return self

    def predict(self, X):
        return self.cov

    def __call__(self, X):
        return self.scale * self.predict(X)

    def score(self, X, diff):
        covs = self.predict(X)
        if covs.ndim == 2:
            return np.einsum('...i,ij,...j->...', diff, np.linalg.inv(covs), diff) / self.scale
        return np.einsum('...i,...ij,...j->...', diff, np.linalg.inv(covs), diff) / self.scale

    def calibrate(self, X, diff, q=0.9, reset=False):
        if reset:
            self.scale = 1.
        n = X.shape[0]
        scores = self.score(X, diff)
        q = min(np.floor((n + 1) * q) / n, 1.)
        self.scale *= np.quantile(scores, q)
        return self


class NearestNeighborsLearner(BaseLearner):
    def __init__(self, dataset_name, seed, n_neighbors=5):
        super().__init__()
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.out_shape = 0
        self.ys = None
        self.dataset_name = dataset_name
        self.seed = seed

    def fit(self, X, diff, **kwargs):
        super().fit(X, diff)
        self.out_shape = diff.shape[-1]
        diff = np.einsum('...i,...j->...ij', diff, diff).reshape((-1, self.out_shape ** 2))
        self.model.fit(X, diff)
        return self

    def predict(self, X):
        covs = self.model.predict(X).reshape(-1, self.out_shape, self.out_shape)
        return 0.95 * covs + 0.05 * super().predict(None)

    def evaluate_and_plot_shapes(self, X, diff, pred, y):
        covs = self.predict(X)
        accuracy = np.mean(self.score(X, diff) <= 1, axis=0)

        det_cov = np.linalg.det(covs)
        average_area = np.mean((np.pi ** (1 / 2)) / math.gamma((1 / 2) + 1) * np.sqrt(det_cov) *
                               (self.scale ** (1 / 2)))

        draw_ellipsoids(covs[:10], self.scale, pred[:10], y[:10], self.dataset_name, self.seed,
                        title=f"NLE uncertainty ellipsoids - accuracy of {round(accuracy, 3)},"
                                       f"average area of {round(average_area, 3)}")

        return accuracy, average_area

    @staticmethod
    def softmax(dist):
        return scipy_softmax(dist, axis=1)
