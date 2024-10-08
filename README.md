# Overview
This project focuses on implementing and evaluating the MVSVRC (Minimal Volume Single-Valued Uncertainty Shape in Ray Cutting), an innovative approach for predicting uncertainty shapes. The aim is to enrich the types of uncertainty shapes that can be predicted, including shapes such as ellipses. The MVSVRC method is evaluated against the NLE (KNN model) to measure its effectiveness.
uncertainty_shape_foregressionproblems_with_2-D_Labels.pdf includes the theory behind this model.


# Main Features
* MVSVRC Implementation:
The core of this project is the development of MVSVRC (Minimal Volume Single-Valued Uncertainty Shape in Ray Cutting), a method designed to predict the shape of uncertainty regions while minimizing their volume. Unlike traditional uncertainty models, which often assume fixed shapes like ellipses or spheres, MVSVRC adapts to the data, producing more flexible uncertainty shapes.

* Comparison with NLE:
The performance of MVSVRC is rigorously evaluated against the Nearest-Neighbor Estimation (NLE) model, which serves as a benchmark. NLE represents a more conventional method of uncertainty estimation based on the distances between nearest neighbors.

* Custom Loss Functions:
MVSVRC and its uncertainty estimation process utilize specially designed loss functions. These functions are crafted to minimize the volume of the predicted uncertainty regions while ensuring they still encapsulate the true values. This tailored optimization approach ensures that the uncertainty predictions are both tight and reliable, improving the overall performance of the model.

* Conformal Prediction:
The project integrates conformal prediction techniques to further enhance uncertainty calibration. Conformal prediction offers a framework to generate valid uncertainty intervals that adapt to the data and provide rigorous probabilistic guarantees. This ensures that the predicted uncertainty regions are not only minimal in volume but also offer accurate coverage, allowing the model to produce confidence intervals with a specified level of certainty.


# Installation
To run this project, you need the following:
* Python 3.8+
* PyTorch
* NumPy
* pandas
* SciPy
* scikit-learn
* xlsxwriter

Clone the current repository. Then, Install the required packages:
pip install -r requirements.txt


# Usage - Running the Models
To run the model, execute the main.py file. You can pass different parameters such as batch size, number of epochs, and uncertainty-related parameters like m, lamda, and u. Here’s an example:
python main.py -batch_size 32 -epochs 150 -lamda 0.02 -m 10 -u 0.9


# Parameters
* batch_size: Batch size for training (default: 32)
* epochs: Number of training epochs for the MVSVRC model (default: 150)
* lamda: Multiplication factor of the shape area term in the uncertainty loss (default: 0.02)
* m: When (2m+1) is the output dimension of the MVSVRC model (default: 10)
* u: Calibration parameter for uncertainty (default: 0.9)


# Output
After running the models, the following outputs will be generated:
* Plots: Uncertainty shape plots for the test and training data are stored in the plots/ directory.
* Model Weights: Trained model weights for the mean and uncertainty models are saved in the mean_weights/ and uncertainty_weights/ directories, respectively.
* Excel Results: Accuracy and uncertainty shape area results are exported to an Excel file model_results.xlsx.


# Results
The model evaluates the accuracy and the average uncertainty shape area for each dataset. The results are saved in the model_results.xlsx file, with the following metrics for each seed and dataset:
* Mean Accuracy
* Standard Deviation of Accuracy
* Mean Uncertainty Shape Area
* Standard Deviation of Uncertainty Shape Area
* The uncertainty shapes are visualized as ellipses or more complex shapes, depending on the model's outputs.
