import math
import os.path
import pandas as pd
import torch
import argparse
from models import SimpleModel
from plots import plot_losses
from data import get_dataset
from learners import MeanLearner, UncertaintyLearner, NearestNeighborsLearner


def export_results_to_excel(results, output_file="results.xlsx"):
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:

        for database_name, seeds_data in results.items():
            models = ['nle', 'mvsvrc']
            data = {model: {'accuracy': [], 'uncertainty_shape_area': []} for model in models}

            for seed, metrics in seeds_data.items():
                for model in models:
                    if model in metrics:
                        data[model]['accuracy'].append(metrics[model]['accuracy'])
                        data[model]['uncertainty_shape_area'].append(metrics[model]['uncertainty_shape_area'])

            result_dict = {}
            for model in models:
                result_dict[model] = {
                    'Mean Accuracy': pd.Series(data[model]['accuracy']).mean(),
                    'Std Accuracy': pd.Series(data[model]['accuracy']).std(),
                    'Mean Uncertainty Shape Area': pd.Series(data[model]['uncertainty_shape_area']).mean(),
                    'Std Uncertainty Shape Area': pd.Series(data[model]['uncertainty_shape_area']).std(),
                }

            result_df = pd.DataFrame(result_dict).T
            result_df.to_excel(writer, sheet_name=database_name)


def fit_and_eval_mvsvrc(X_train, Y_train, X_calibration, Y_calibration, X_test, Y_test, results,
                        input_size, hidden_size, epochs, batch_size, mean_model, lamda, m, u, database_name, seed):
    uncertainty_model = SimpleModel(input_size + 2, hidden_size, output_size=(2 * m + 1), change_init_weight=True)
    uncertainty_model_path = f'uncertainty_weights/uncertainty_model_weights_{database_name}_{seed}.pth'

    uncertainty_learner = UncertaintyLearner(epochs, batch_size, uncertainty_model, mean_model, lamda,
                                             uncertainty_model_path, database_name, seed)

    if not os.path.exists(uncertainty_model_path):
        uncertainty_train_loss, uncertainty_test_loss = uncertainty_learner.fit(X_train, X_test, Y_train, Y_test)
        plot_losses(uncertainty_train_loss, uncertainty_test_loss, database_name, seed, plot_title="MVSVRC Model")
    else:
        uncertainty_model.load_state_dict(torch.load(uncertainty_model_path))

    uncertainty_learner.calibrate(X_calibration, Y_calibration, u)

    accuracy, uncertainty_shape_area = uncertainty_learner.evaluate_and_plot_shapes(X_test, Y_test)
    results[database_name][seed]["mvsvrc"] = {'accuracy': accuracy, 'uncertainty_shape_area': uncertainty_shape_area}


def fit_and_eval_nle(X_train, Y_train, X_calibration, Y_calibration, X_test, Y_test, results,
                        mean_model, u, database_name, seed):
    pred_train = mean_model(torch.tensor(X_train)).detach().numpy()
    pred_calibration = mean_model(torch.tensor(X_calibration)).detach().numpy()
    pred_test = mean_model(torch.tensor(X_test)).detach().numpy()
    diff_train = pred_train - Y_train
    diff_calibration = pred_calibration - Y_calibration
    diff_test = pred_test - Y_test

    # based on https://copa-conference.com/papers/COPA2022_paper_7.pdf
    nle_learner = NearestNeighborsLearner(database_name, seed, n_neighbors=X_train.shape[0] // 20)
    nle_learner.fit(X_train, diff_train)
    nle_learner.calibrate(X_calibration, diff_calibration, u)

    accuracy, uncertainty_shape_area = nle_learner.evaluate_and_plot_shapes(X_test, diff_test, pred_test, Y_test)
    results[database_name][seed]["nle"] = {'accuracy': accuracy, 'uncertainty_shape_area': uncertainty_shape_area}


def main(database_name, seed, batch_size, epochs, m, lamda, u, results, test_ratio=0.1, calibration_ratio=0.15):
    X_train, Y_train, X_calibration, Y_calibration, X_test, Y_test = get_dataset(
        database_name, datasets_folder='datasets', seed=seed, test_ratio=test_ratio, calibration_ratio=calibration_ratio)
    input_size = X_train.shape[1]
    hidden_size = math.ceil(math.sqrt(X_train.shape[0] / 2))
    mean_model = SimpleModel(input_size, hidden_size)

    mean_model_path = f'mean_weights/mean_model_weights_{database_name}_{seed}.pth'

    if not os.path.exists(mean_model_path):
        mean_learner = MeanLearner(100, batch_size, mean_model, mean_model_path)
        mean_train_loss, mean_test_loss = mean_learner.fit(X_train, X_test, Y_train, Y_test)
        plot_losses(mean_train_loss, mean_test_loss, database_name, seed, plot_title="Mean Model")
    else:
        mean_model.load_state_dict(torch.load(mean_model_path))

    fit_and_eval_nle(X_train, Y_train, X_calibration, Y_calibration, X_test, Y_test, results,
                        mean_model, u, database_name, seed)
    fit_and_eval_mvsvrc(X_train, Y_train, X_calibration, Y_calibration, X_test, Y_test, results,
                        input_size, hidden_size, epochs, batch_size, mean_model, lamda, m, u, database_name, seed)


if __name__ == '__main__':
    for directory in ['plots', 'mean_weights', 'uncertainty_weights']:
        if not os.path.exists(directory):
            os.makedirs(directory)

    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=32, required=False)
    parser.add_argument('-epochs', type=int, default=150, required=False)
    parser.add_argument('-lamda', type=float, default=0.02, required=False)
    parser.add_argument('-m', type=int, default=10, required=False)
    parser.add_argument('-u', type=int, default=0.9, required=False)
    args = parser.parse_args()

    results = dict()
    for database in ["ble_rssi", "enb", "residential_building"]:
        results[database] = dict()
        for i in range(10):
            results[database][i] = dict()
            main(database_name=database, seed=i, batch_size=args.batch_size, epochs=args.epochs, m=args.m,
             lamda=args.lamda, u=args.u, results=results)

    export_results_to_excel(results, output_file="model_results.xlsx")
